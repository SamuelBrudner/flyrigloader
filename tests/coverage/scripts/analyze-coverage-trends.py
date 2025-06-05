#!/usr/bin/env python3
"""
FlyrigLoader Coverage Trend Analysis Script

Comprehensive coverage trend analysis providing automated tracking of coverage metrics 
over time with regression detection, performance correlation analysis, and statistical 
trend validation. Implements comprehensive historical analysis enabling proactive 
quality assurance per Section 0.2.5 infrastructure requirements.

This script serves as the central orchestrator for coverage trend monitoring in the 
flyrigloader test suite enhancement system, providing:

- Historical coverage data persistence and trend tracking over time
- Statistical regression detection with configurable sensitivity thresholds
- Performance benchmark correlation analysis for comprehensive quality metrics
- Automated alerting for coverage degradation with actionable feedback
- Confidence interval analysis and significance testing for reliable trend detection
- CI/CD integration with automated trend monitoring and quality gate enforcement

Requirements Implementation:
- Section 0.2.5: Coverage trend tracking over time per Infrastructure Updates
- TST-COV-004: Block merges when coverage drops below thresholds
- TST-PERF-001: Data loading SLA validation within 1s per 100MB
- Section 3.6.4: Quality metrics dashboard integration with coverage trend tracking
- Section 4.1.1.5: Test execution workflow with automated quality gate enforcement
- Section 2.1.12: Coverage Enhancement System with detailed reporting and visualization

Author: FlyrigLoader Test Suite Enhancement Team
Created: 2024-12-19
License: MIT
"""

import argparse
import json
import logging
import os
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports for statistical analysis and visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for CI/CD environments
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Import project dependencies
try:
    from .generate_coverage_reports import CoverageReportGenerator
except ImportError:
    try:
        from generate_coverage_reports import CoverageReportGenerator
    except ImportError:
        print("ERROR: Unable to import CoverageReportGenerator")
        print("Ensure generate-coverage-reports.py is available in the same directory")
        sys.exit(1)


class CoverageTrendAnalyzer:
    """
    Comprehensive coverage trend analysis engine implementing automated tracking
    of coverage metrics over time with advanced statistical analysis, regression
    detection, and performance correlation capabilities.
    
    This class orchestrates the complete coverage trend analysis pipeline including:
    - Historical coverage data collection and persistence
    - Statistical trend analysis with confidence intervals
    - Regression detection algorithms with configurable sensitivity
    - Performance benchmark correlation and SLA validation
    - Automated alerting and notification systems
    - CI/CD integration with quality gate enforcement
    """

    def __init__(self,
                 coverage_data_file: Optional[str] = None,
                 thresholds_file: Optional[str] = None,
                 quality_gates_file: Optional[str] = None,
                 historical_data_dir: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 verbose: bool = False):
        """
        Initialize the coverage trend analyzer with configuration and data sources.
        
        Args:
            coverage_data_file: Path to current coverage data file (.coverage)
            thresholds_file: Path to coverage thresholds JSON file
            quality_gates_file: Path to quality gates YAML file
            historical_data_dir: Directory for storing historical trend data
            output_dir: Base output directory for generated reports and visualizations
            verbose: Enable verbose logging output
        """
        self.verbose = verbose
        self.setup_logging()
        
        # Initialize paths and configuration
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.coverage_dir = self.project_root / "tests" / "coverage"
        self.scripts_dir = self.coverage_dir / "scripts"
        
        # Set default paths if not provided
        self.coverage_data_file = coverage_data_file or str(self.project_root / ".coverage")
        self.thresholds_file = thresholds_file or str(self.coverage_dir / "coverage-thresholds.json")
        self.quality_gates_file = quality_gates_file or str(self.coverage_dir / "quality-gates.yml")
        self.historical_data_dir = Path(historical_data_dir) if historical_data_dir else self.coverage_dir / "historical"
        self.output_dir = Path(output_dir) if output_dir else self.coverage_dir / "trends"
        
        # Ensure directories exist
        self.historical_data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize configuration and data storage
        self.thresholds: Dict[str, Any] = {}
        self.quality_gates: Dict[str, Any] = {}
        self.current_coverage: Dict[str, Any] = {}
        self.historical_data: List[Dict[str, Any]] = []
        self.trend_analysis: Dict[str, Any] = {}
        
        # Runtime statistics and analysis settings
        self.start_time = time.time()
        self.analysis_stats: Dict[str, Any] = {
            'start_time': datetime.now(timezone.utc),
            'analysis_completed': False,
            'warnings': [],
            'errors': [],
            'quality_gates_passed': False,
            'regression_detected': False
        }
        
        # Statistical analysis configuration
        self.confidence_level = 0.95  # 95% confidence interval
        self.regression_sensitivity = 0.05  # 5% decrease threshold
        self.trend_window_days = 30  # 30-day trend analysis window
        self.minimum_data_points = 3  # Minimum data points for statistical analysis
        
        self.logger.info(f"Initialized CoverageTrendAnalyzer")
        self.logger.info(f"Project root: {self.project_root}")
        self.logger.info(f"Historical data directory: {self.historical_data_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")

    def setup_logging(self) -> None:
        """Configure comprehensive logging for coverage trend analysis."""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        log_file = self.project_root / "tests" / "coverage" / "coverage-trends.log"
        
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file, mode='a')
            ]
        )
        self.logger = logging.getLogger("flyrigloader.coverage.trends")

    def load_configuration(self) -> None:
        """
        Load comprehensive configuration from JSON and YAML files including
        coverage thresholds, quality gates, and trend analysis settings.
        
        Raises:
            FileNotFoundError: If configuration files are missing
            json.JSONDecodeError: If JSON configuration files are malformed
        """
        self.logger.info("Loading configuration files...")
        
        # Load coverage thresholds
        try:
            with open(self.thresholds_file, 'r', encoding='utf-8') as f:
                self.thresholds = json.load(f)
            self.logger.info(f"Loaded coverage thresholds from {self.thresholds_file}")
        except FileNotFoundError:
            self.logger.error(f"Coverage thresholds file not found: {self.thresholds_file}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in thresholds config: {e}")
            raise

        # Load quality gates configuration
        try:
            import yaml
            with open(self.quality_gates_file, 'r', encoding='utf-8') as f:
                self.quality_gates = yaml.safe_load(f)
            self.logger.info(f"Loaded quality gates from {self.quality_gates_file}")
        except ImportError:
            self.logger.warning("PyYAML not available, loading quality gates as JSON")
            try:
                # Fallback to JSON parsing if YAML not available
                with open(self.quality_gates_file.replace('.yml', '.json'), 'r', encoding='utf-8') as f:
                    self.quality_gates = json.load(f)
            except FileNotFoundError:
                self.logger.error("Neither YAML nor JSON quality gates file found")
                raise
        except FileNotFoundError:
            self.logger.error(f"Quality gates file not found: {self.quality_gates_file}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load quality gates: {e}")
            raise

        # Validate configuration structure
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate that required configuration keys are present and properly formatted."""
        # Validate thresholds configuration
        required_threshold_keys = ['global_settings', 'critical_modules']
        for key in required_threshold_keys:
            if key not in self.thresholds:
                raise ValueError(f"Missing required threshold key: {key}")

        # Validate quality gates configuration
        required_quality_keys = ['coverage', 'performance', 'global']
        for key in required_quality_keys:
            if key not in self.quality_gates:
                raise ValueError(f"Missing required quality gates key: {key}")

        # Extract trend analysis configuration
        trend_config = self.quality_gates.get('coverage', {}).get('regression_detection', {})
        if trend_config.get('enabled', True):
            self.regression_sensitivity = trend_config.get('maximum_allowed_decrease', 0.05)
            self.trend_window_days = trend_config.get('trend_analysis_window', 30)

        self.logger.info("Configuration validation completed successfully")
        self.logger.info(f"Regression sensitivity: {self.regression_sensitivity}")
        self.logger.info(f"Trend analysis window: {self.trend_window_days} days")

    def collect_current_coverage_data(self) -> None:
        """
        Collect current coverage metrics using the CoverageReportGenerator
        and extract comprehensive coverage statistics for trend analysis.
        
        Raises:
            RuntimeError: If coverage data collection fails
        """
        self.logger.info("Collecting current coverage data...")
        
        try:
            # Initialize coverage report generator
            generator = CoverageReportGenerator(
                coverage_data_file=self.coverage_data_file,
                thresholds_file=self.thresholds_file,
                verbose=self.verbose
            )
            
            # Load configuration and data
            generator.load_configuration()
            generator.load_coverage_data()
            
            # Perform coverage analysis
            analysis_results = generator.analyze_coverage_data()
            
            # Extract key metrics for trend analysis
            self.current_coverage = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'summary': analysis_results['summary'],
                'quality_gates': analysis_results['quality_gates'],
                'statistics': analysis_results['statistics'],
                'critical_modules': analysis_results['critical_modules'],
                'module_coverage': {}
            }
            
            # Extract module-specific coverage data
            for module_path, module_data in analysis_results['modules'].items():
                self.current_coverage['module_coverage'][module_path] = {
                    'line_coverage': module_data['coverage_percentage'],
                    'branch_coverage': module_data['branch_coverage_percentage'],
                    'category': module_data['category'],
                    'status': module_data['status'],
                    'statements': module_data['statements'],
                    'missing': module_data['missing']
                }
            
            # Add build and environment metadata
            self.current_coverage['metadata'] = self._collect_build_metadata()
            
            self.logger.info(f"Current overall coverage: {self.current_coverage['summary']['overall_coverage_percentage']:.2f}%")
            self.logger.info(f"Current branch coverage: {self.current_coverage['summary']['branch_coverage_percentage']:.2f}%")
            
        except Exception as e:
            self.logger.error(f"Failed to collect current coverage data: {e}")
            raise RuntimeError(f"Coverage data collection failed: {e}")

    def _collect_build_metadata(self) -> Dict[str, Any]:
        """Collect build and environment metadata for historical tracking."""
        metadata = {
            'collection_time': datetime.now(timezone.utc).isoformat(),
            'python_version': sys.version.split()[0],
            'platform': sys.platform,
        }
        
        # Add CI/CD metadata if available
        ci_vars = {
            'build_number': 'BUILD_NUMBER',
            'build_url': 'BUILD_URL',
            'branch_name': 'BRANCH_NAME',
            'commit_sha': 'COMMIT_SHA',
            'pr_number': 'PR_NUMBER',
            'ci_environment': 'CI_ENVIRONMENT',
            'runner_os': 'RUNNER_OS'
        }
        
        for key, env_var in ci_vars.items():
            metadata[key] = os.environ.get(env_var, 'unknown')
        
        # Add Git information if available
        try:
            import git
            repo = git.Repo(self.project_root)
            metadata.update({
                'git_branch': repo.active_branch.name,
                'git_commit': repo.head.commit.hexsha[:8],
                'git_commit_message': repo.head.commit.message.strip(),
                'git_is_dirty': repo.is_dirty()
            })
        except Exception:
            self.logger.debug("Git information not available")
        
        return metadata

    def load_historical_data(self) -> None:
        """
        Load historical coverage data from persistent storage for trend analysis.
        Creates historical data structure if it doesn't exist.
        """
        self.logger.info("Loading historical coverage data...")
        
        historical_file = self.historical_data_dir / "coverage_history.json"
        
        if historical_file.exists():
            try:
                with open(historical_file, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                # Validate and filter historical data
                self.historical_data = self._validate_historical_data(raw_data.get('entries', []))
                
                self.logger.info(f"Loaded {len(self.historical_data)} historical data points")
                
                if self.historical_data:
                    latest_entry = max(self.historical_data, key=lambda x: x['timestamp'])
                    self.logger.info(f"Latest entry: {latest_entry['timestamp']}")
                
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.error(f"Failed to load historical data: {e}")
                self.historical_data = []
        else:
            self.logger.info("No historical data found, starting fresh")
            self.historical_data = []

    def _validate_historical_data(self, raw_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and filter historical data entries to ensure data quality.
        
        Args:
            raw_entries: List of raw historical data entries
            
        Returns:
            List of validated and filtered historical data entries
        """
        validated_entries = []
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=365)  # Keep last year
        
        for entry in raw_entries:
            try:
                # Validate required fields
                if not all(key in entry for key in ['timestamp', 'summary']):
                    continue
                
                # Parse and validate timestamp
                entry_time = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                
                # Filter out entries older than cutoff
                if entry_time < cutoff_date:
                    continue
                
                # Validate coverage data structure
                if 'overall_coverage_percentage' not in entry['summary']:
                    continue
                
                validated_entries.append(entry)
                
            except (ValueError, KeyError, TypeError) as e:
                self.logger.warning(f"Skipping invalid historical entry: {e}")
                continue
        
        # Sort by timestamp
        validated_entries.sort(key=lambda x: x['timestamp'])
        
        return validated_entries

    def persist_current_data(self) -> None:
        """
        Persist current coverage data to historical storage for future trend analysis.
        Implements data deduplication and storage optimization.
        """
        self.logger.info("Persisting current coverage data to historical storage...")
        
        # Add current data to historical collection
        self.historical_data.append(self.current_coverage.copy())
        
        # Remove duplicates and old entries
        self.historical_data = self._deduplicate_historical_data(self.historical_data)
        
        # Prepare data for storage
        storage_data = {
            'metadata': {
                'version': '2.0',
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'total_entries': len(self.historical_data),
                'data_schema': 'flyrigloader_coverage_trends_v2'
            },
            'entries': self.historical_data
        }
        
        # Write to historical storage
        historical_file = self.historical_data_dir / "coverage_history.json"
        try:
            with open(historical_file, 'w', encoding='utf-8') as f:
                json.dump(storage_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Persisted {len(self.historical_data)} entries to {historical_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to persist historical data: {e}")
            raise

    def _deduplicate_historical_data(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate entries and implement data retention policies.
        
        Args:
            entries: List of historical data entries
            
        Returns:
            Deduplicated and filtered list of entries
        """
        # Sort by timestamp
        entries.sort(key=lambda x: x['timestamp'])
        
        # Remove exact duplicates based on timestamp and coverage
        seen_signatures = set()
        deduplicated = []
        
        for entry in entries:
            signature = (
                entry['timestamp'],
                entry['summary']['overall_coverage_percentage'],
                entry['summary']['branch_coverage_percentage']
            )
            
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                deduplicated.append(entry)
        
        # Implement retention policy (keep last 500 entries)
        max_entries = 500
        if len(deduplicated) > max_entries:
            self.logger.info(f"Trimming historical data to {max_entries} entries")
            deduplicated = deduplicated[-max_entries:]
        
        return deduplicated

    def perform_trend_analysis(self) -> None:
        """
        Perform comprehensive statistical trend analysis including regression detection,
        confidence interval calculation, and performance correlation analysis.
        """
        self.logger.info("Performing comprehensive trend analysis...")
        
        if len(self.historical_data) < self.minimum_data_points:
            self.logger.warning(f"Insufficient data for trend analysis ({len(self.historical_data)} < {self.minimum_data_points})")
            self.trend_analysis = self._create_minimal_analysis()
            return
        
        analysis_start = time.time()
        
        # Initialize trend analysis results
        self.trend_analysis = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'data_points': len(self.historical_data),
            'analysis_window_days': self.trend_window_days,
            'overall_trends': {},
            'module_trends': {},
            'regression_analysis': {},
            'statistical_metrics': {},
            'performance_correlation': {},
            'quality_predictions': {},
            'alerts': []
        }
        
        # Perform overall coverage trend analysis
        self._analyze_overall_trends()
        
        # Perform module-specific trend analysis
        self._analyze_module_trends()
        
        # Perform regression detection
        self._detect_coverage_regression()
        
        # Calculate statistical metrics
        self._calculate_statistical_metrics()
        
        # Correlate with performance data if available
        self._correlate_performance_data()
        
        # Generate quality predictions
        self._generate_quality_predictions()
        
        # Generate alerts for detected issues
        self._generate_trend_alerts()
        
        analysis_duration = time.time() - analysis_start
        self.trend_analysis['analysis_duration'] = analysis_duration
        
        self.logger.info(f"Trend analysis completed in {analysis_duration:.2f} seconds")
        self.logger.info(f"Analyzed {len(self.historical_data)} data points over {self.trend_window_days} days")

    def _create_minimal_analysis(self) -> Dict[str, Any]:
        """Create minimal analysis structure when insufficient data is available."""
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'data_points': len(self.historical_data),
            'insufficient_data': True,
            'minimum_required': self.minimum_data_points,
            'overall_trends': {'trend_available': False},
            'regression_analysis': {'regression_detected': False, 'confidence': 'low'},
            'alerts': [{
                'type': 'warning',
                'message': f'Insufficient historical data for trend analysis ({len(self.historical_data)} < {self.minimum_data_points})',
                'severity': 'medium',
                'actionable': True,
                'recommendations': ['Collect more coverage data over time', 'Run regular coverage analysis']
            }]
        }

    def _analyze_overall_trends(self) -> None:
        """Analyze overall coverage trends with statistical significance testing."""
        self.logger.debug("Analyzing overall coverage trends...")
        
        # Extract time series data
        timestamps = [datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00')) for entry in self.historical_data]
        overall_coverage = [entry['summary']['overall_coverage_percentage'] for entry in self.historical_data]
        branch_coverage = [entry['summary']['branch_coverage_percentage'] for entry in self.historical_data]
        
        # Filter data to analysis window
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.trend_window_days)
        window_data = [(t, oc, bc) for t, oc, bc in zip(timestamps, overall_coverage, branch_coverage) if t >= cutoff_date]
        
        if len(window_data) < 2:
            self.trend_analysis['overall_trends'] = {'insufficient_window_data': True}
            return
        
        window_timestamps, window_overall, window_branch = zip(*window_data)
        
        # Calculate trend metrics
        self.trend_analysis['overall_trends'] = {
            'line_coverage': self._calculate_trend_metrics(list(window_timestamps), list(window_overall)),
            'branch_coverage': self._calculate_trend_metrics(list(window_timestamps), list(window_branch)),
            'data_points_in_window': len(window_data),
            'window_start': min(window_timestamps).isoformat(),
            'window_end': max(window_timestamps).isoformat()
        }

    def _calculate_trend_metrics(self, timestamps: List[datetime], values: List[float]) -> Dict[str, Any]:
        """
        Calculate comprehensive trend metrics including slope, correlation, and significance.
        
        Args:
            timestamps: List of timestamp values
            values: List of coverage percentage values
            
        Returns:
            Dictionary containing trend analysis metrics
        """
        if len(values) < 2:
            return {'insufficient_data': True}
        
        # Convert timestamps to numeric values for correlation analysis
        epoch_times = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
        
        trend_metrics = {
            'current_value': values[-1],
            'previous_value': values[-2] if len(values) > 1 else values[0],
            'min_value': min(values),
            'max_value': max(values),
            'mean_value': statistics.mean(values),
            'median_value': statistics.median(values),
            'std_deviation': statistics.stdev(values) if len(values) > 1 else 0.0,
            'value_change': values[-1] - values[0],
            'percentage_change': ((values[-1] - values[0]) / values[0] * 100) if values[0] > 0 else 0.0
        }
        
        # Calculate linear regression if scipy is available
        if SCIPY_AVAILABLE and len(values) > 2:
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(epoch_times, values)
                
                trend_metrics.update({
                    'linear_trend': {
                        'slope': slope,
                        'intercept': intercept,
                        'correlation_coefficient': r_value,
                        'p_value': p_value,
                        'standard_error': std_err,
                        'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
                        'trend_strength': abs(r_value),
                        'statistically_significant': p_value < 0.05
                    }
                })
                
                # Calculate confidence intervals
                if len(values) > 3:
                    confidence_interval = self._calculate_confidence_interval(values, self.confidence_level)
                    trend_metrics['confidence_interval'] = confidence_interval
                    
            except Exception as e:
                self.logger.warning(f"Failed to calculate linear regression: {e}")
        
        # Detect trend patterns
        trend_metrics['trend_pattern'] = self._detect_trend_pattern(values)
        
        return trend_metrics

    def _calculate_confidence_interval(self, values: List[float], confidence_level: float) -> Dict[str, float]:
        """
        Calculate confidence interval for coverage values.
        
        Args:
            values: List of coverage values
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Dictionary containing confidence interval bounds
        """
        if not SCIPY_AVAILABLE or len(values) < 3:
            return {'error': 'Insufficient data or scipy not available'}
        
        try:
            mean_val = statistics.mean(values)
            std_err = statistics.stdev(values) / (len(values) ** 0.5)
            
            # Calculate t-distribution critical value
            alpha = 1 - confidence_level
            degrees_freedom = len(values) - 1
            t_critical = stats.t.ppf(1 - alpha/2, degrees_freedom)
            
            margin_error = t_critical * std_err
            
            return {
                'mean': mean_val,
                'margin_of_error': margin_error,
                'lower_bound': mean_val - margin_error,
                'upper_bound': mean_val + margin_error,
                'confidence_level': confidence_level
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate confidence interval: {e}")
            return {'error': str(e)}

    def _detect_trend_pattern(self, values: List[float]) -> Dict[str, Any]:
        """
        Detect trend patterns in coverage data.
        
        Args:
            values: List of coverage values
            
        Returns:
            Dictionary describing the detected trend pattern
        """
        if len(values) < 3:
            return {'pattern': 'insufficient_data'}
        
        # Calculate moving averages for pattern detection
        window_size = min(5, len(values) // 2)
        if window_size < 2:
            return {'pattern': 'too_few_points'}
        
        moving_avg = []
        for i in range(len(values) - window_size + 1):
            avg = sum(values[i:i + window_size]) / window_size
            moving_avg.append(avg)
        
        # Analyze pattern characteristics
        if len(moving_avg) < 2:
            return {'pattern': 'stable'}
        
        # Check for consistent trends
        increases = sum(1 for i in range(1, len(moving_avg)) if moving_avg[i] > moving_avg[i-1])
        decreases = sum(1 for i in range(1, len(moving_avg)) if moving_avg[i] < moving_avg[i-1])
        stable = len(moving_avg) - 1 - increases - decreases
        
        total_changes = len(moving_avg) - 1
        
        pattern_analysis = {
            'increases': increases,
            'decreases': decreases,
            'stable_periods': stable,
            'volatility': statistics.stdev(values) if len(values) > 1 else 0.0,
            'range_span': max(values) - min(values)
        }
        
        # Classify pattern
        if increases > total_changes * 0.7:
            pattern_analysis['pattern'] = 'consistently_increasing'
        elif decreases > total_changes * 0.7:
            pattern_analysis['pattern'] = 'consistently_decreasing'
        elif stable > total_changes * 0.7:
            pattern_analysis['pattern'] = 'stable'
        elif pattern_analysis['volatility'] > 5.0:
            pattern_analysis['pattern'] = 'volatile'
        else:
            pattern_analysis['pattern'] = 'mixed'
        
        return pattern_analysis

    def _analyze_module_trends(self) -> None:
        """Analyze coverage trends for individual modules and categories."""
        self.logger.debug("Analyzing module-specific coverage trends...")
        
        module_trends = {}
        
        # Extract all unique modules from historical data
        all_modules = set()
        for entry in self.historical_data:
            if 'module_coverage' in entry:
                all_modules.update(entry['module_coverage'].keys())
        
        # Analyze trends for each module
        for module_path in all_modules:
            module_data = []
            timestamps = []
            
            # Extract time series for this module
            for entry in self.historical_data:
                if 'module_coverage' in entry and module_path in entry['module_coverage']:
                    timestamps.append(datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00')))
                    module_data.append(entry['module_coverage'][module_path]['line_coverage'])
            
            if len(module_data) >= 2:
                # Filter to analysis window
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.trend_window_days)
                window_data = [(t, d) for t, d in zip(timestamps, module_data) if t >= cutoff_date]
                
                if len(window_data) >= 2:
                    window_timestamps, window_values = zip(*window_data)
                    
                    module_trends[module_path] = {
                        'trend_metrics': self._calculate_trend_metrics(list(window_timestamps), list(window_values)),
                        'category': self._get_module_category(module_path),
                        'data_points': len(window_data),
                        'is_critical': self._is_critical_module(module_path)
                    }
        
        self.trend_analysis['module_trends'] = module_trends

    def _get_module_category(self, module_path: str) -> str:
        """Determine module category based on flyrigloader architecture."""
        if 'api.py' in module_path:
            return 'api_layer'
        elif '/config/' in module_path:
            return 'configuration_system'
        elif '/discovery/' in module_path:
            return 'discovery_engine'
        elif '/io/' in module_path:
            return 'io_pipeline'
        elif '/utils/' in module_path:
            return 'utilities'
        elif '__init__.py' in module_path:
            return 'initialization_modules'
        else:
            return 'other'

    def _is_critical_module(self, module_path: str) -> bool:
        """Check if module is classified as critical requiring 100% coverage."""
        critical_modules = self.thresholds.get('critical_modules', {}).get('modules', {})
        
        # Check direct path match
        for critical_path in critical_modules.keys():
            if module_path in critical_path or critical_path in module_path:
                return True
        
        # Check category-based classification
        category = self._get_module_category(module_path)
        critical_categories = ['api_layer', 'configuration_system', 'discovery_engine', 'io_pipeline']
        
        return category in critical_categories

    def _detect_coverage_regression(self) -> None:
        """
        Detect coverage regression using statistical analysis and configurable thresholds.
        Implements TST-COV-004 requirement to block merges when coverage drops below thresholds.
        """
        self.logger.debug("Detecting coverage regression...")
        
        regression_analysis = {
            'regression_detected': False,
            'severity': 'none',
            'confidence': 'low',
            'affected_modules': [],
            'overall_regression': {},
            'critical_module_regression': {},
            'threshold_violations': []
        }
        
        if len(self.historical_data) < 2:
            regression_analysis['insufficient_data'] = True
            self.trend_analysis['regression_analysis'] = regression_analysis
            return
        
        # Analyze overall coverage regression
        overall_regression = self._analyze_overall_regression()
        regression_analysis['overall_regression'] = overall_regression
        
        if overall_regression['regression_detected']:
            regression_analysis['regression_detected'] = True
            regression_analysis['severity'] = max(regression_analysis['severity'], overall_regression['severity'], key=self._severity_level)
        
        # Analyze critical module regression
        critical_regression = self._analyze_critical_module_regression()
        regression_analysis['critical_module_regression'] = critical_regression
        
        if critical_regression['regression_detected']:
            regression_analysis['regression_detected'] = True
            regression_analysis['severity'] = max(regression_analysis['severity'], critical_regression['severity'], key=self._severity_level)
            regression_analysis['affected_modules'].extend(critical_regression['affected_modules'])
        
        # Check threshold violations
        threshold_violations = self._check_threshold_violations()
        regression_analysis['threshold_violations'] = threshold_violations
        
        if threshold_violations:
            regression_analysis['regression_detected'] = True
            regression_analysis['severity'] = max(regression_analysis['severity'], 'high', key=self._severity_level)
        
        # Calculate overall confidence
        regression_analysis['confidence'] = self._calculate_regression_confidence(regression_analysis)
        
        # Update analysis stats
        self.analysis_stats['regression_detected'] = regression_analysis['regression_detected']
        
        self.trend_analysis['regression_analysis'] = regression_analysis

    def _severity_level(self, severity: str) -> int:
        """Convert severity string to numeric level for comparison."""
        levels = {'none': 0, 'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        return levels.get(severity, 0)

    def _analyze_overall_regression(self) -> Dict[str, Any]:
        """Analyze regression in overall coverage metrics."""
        recent_entries = self.historical_data[-5:]  # Last 5 entries
        
        if len(recent_entries) < 2:
            return {'regression_detected': False, 'insufficient_data': True}
        
        current_coverage = recent_entries[-1]['summary']['overall_coverage_percentage']
        baseline_coverage = statistics.mean([entry['summary']['overall_coverage_percentage'] for entry in recent_entries[:-1]])
        
        coverage_drop = baseline_coverage - current_coverage
        percentage_drop = (coverage_drop / baseline_coverage * 100) if baseline_coverage > 0 else 0
        
        # Check against sensitivity threshold
        regression_detected = percentage_drop > (self.regression_sensitivity * 100)
        
        severity = 'none'
        if regression_detected:
            if percentage_drop > 10:
                severity = 'critical'
            elif percentage_drop > 5:
                severity = 'high'
            elif percentage_drop > 2:
                severity = 'medium'
            else:
                severity = 'low'
        
        return {
            'regression_detected': regression_detected,
            'current_coverage': current_coverage,
            'baseline_coverage': baseline_coverage,
            'coverage_drop': coverage_drop,
            'percentage_drop': percentage_drop,
            'severity': severity,
            'threshold_exceeded': percentage_drop > (self.regression_sensitivity * 100)
        }

    def _analyze_critical_module_regression(self) -> Dict[str, Any]:
        """Analyze regression in critical modules requiring 100% coverage."""
        critical_regression = {
            'regression_detected': False,
            'affected_modules': [],
            'severity': 'none'
        }
        
        if 'module_coverage' not in self.current_coverage:
            return critical_regression
        
        current_modules = self.current_coverage['module_coverage']
        
        for module_path, module_data in current_modules.items():
            if self._is_critical_module(module_path):
                current_coverage = module_data['line_coverage']
                
                # Critical modules must maintain 100% coverage
                if current_coverage < 100.0:
                    critical_regression['regression_detected'] = True
                    critical_regression['affected_modules'].append({
                        'module': module_path,
                        'current_coverage': current_coverage,
                        'required_coverage': 100.0,
                        'coverage_gap': 100.0 - current_coverage
                    })
                    critical_regression['severity'] = 'critical'
        
        return critical_regression

    def _check_threshold_violations(self) -> List[Dict[str, Any]]:
        """Check for threshold violations based on quality gates configuration."""
        violations = []
        
        # Check overall threshold
        overall_threshold = self.quality_gates.get('coverage', {}).get('overall_coverage_threshold', 90.0)
        current_overall = self.current_coverage['summary']['overall_coverage_percentage']
        
        if current_overall < overall_threshold:
            violations.append({
                'type': 'overall_coverage',
                'threshold': overall_threshold,
                'actual': current_overall,
                'gap': overall_threshold - current_overall,
                'severity': 'high'
            })
        
        # Check branch coverage threshold
        branch_threshold = self.quality_gates.get('coverage', {}).get('overall_branch_threshold', 90.0)
        current_branch = self.current_coverage['summary']['branch_coverage_percentage']
        
        if current_branch < branch_threshold:
            violations.append({
                'type': 'branch_coverage',
                'threshold': branch_threshold,
                'actual': current_branch,
                'gap': branch_threshold - current_branch,
                'severity': 'high'
            })
        
        # Check critical module thresholds
        critical_threshold = self.quality_gates.get('coverage', {}).get('critical_module_coverage_threshold', 100.0)
        
        for module_path, module_data in self.current_coverage.get('module_coverage', {}).items():
            if self._is_critical_module(module_path):
                module_coverage = module_data['line_coverage']
                if module_coverage < critical_threshold:
                    violations.append({
                        'type': 'critical_module_coverage',
                        'module': module_path,
                        'threshold': critical_threshold,
                        'actual': module_coverage,
                        'gap': critical_threshold - module_coverage,
                        'severity': 'critical'
                    })
        
        return violations

    def _calculate_regression_confidence(self, regression_analysis: Dict[str, Any]) -> str:
        """Calculate confidence level for regression detection based on data quality and statistical metrics."""
        if regression_analysis.get('insufficient_data'):
            return 'low'
        
        confidence_factors = []
        
        # Data quantity factor
        data_points = len(self.historical_data)
        if data_points >= 10:
            confidence_factors.append(0.9)
        elif data_points >= 5:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.3)
        
        # Regression severity factor
        severity = regression_analysis.get('severity', 'none')
        severity_confidence = {'none': 0.1, 'low': 0.5, 'medium': 0.7, 'high': 0.9, 'critical': 1.0}
        confidence_factors.append(severity_confidence.get(severity, 0.1))
        
        # Statistical significance factor
        overall_regression = regression_analysis.get('overall_regression', {})
        if 'threshold_exceeded' in overall_regression and overall_regression['threshold_exceeded']:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.4)
        
        # Calculate weighted average confidence
        avg_confidence = sum(confidence_factors) / len(confidence_factors)
        
        if avg_confidence >= 0.8:
            return 'high'
        elif avg_confidence >= 0.6:
            return 'medium'
        else:
            return 'low'

    def _calculate_statistical_metrics(self) -> None:
        """Calculate comprehensive statistical metrics for trend analysis."""
        self.logger.debug("Calculating statistical metrics...")
        
        if len(self.historical_data) < 2:
            self.trend_analysis['statistical_metrics'] = {'insufficient_data': True}
            return
        
        # Extract coverage time series
        overall_coverage = [entry['summary']['overall_coverage_percentage'] for entry in self.historical_data]
        branch_coverage = [entry['summary']['branch_coverage_percentage'] for entry in self.historical_data]
        
        statistical_metrics = {
            'overall_coverage_stats': self._calculate_descriptive_stats(overall_coverage),
            'branch_coverage_stats': self._calculate_descriptive_stats(branch_coverage),
            'data_quality_metrics': self._calculate_data_quality_metrics(),
            'trend_stability': self._calculate_trend_stability(overall_coverage),
            'prediction_reliability': self._calculate_prediction_reliability()
        }
        
        self.trend_analysis['statistical_metrics'] = statistical_metrics

    def _calculate_descriptive_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate descriptive statistics for a series of values."""
        if not values:
            return {'error': 'no_data'}
        
        stats_dict = {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'mode': statistics.mode(values) if len(set(values)) < len(values) else None,
            'min': min(values),
            'max': max(values),
            'range': max(values) - min(values),
            'variance': statistics.variance(values) if len(values) > 1 else 0.0,
            'std_deviation': statistics.stdev(values) if len(values) > 1 else 0.0
        }
        
        # Calculate percentiles if numpy is available
        if NUMPY_AVAILABLE:
            try:
                np_values = np.array(values)
                stats_dict.update({
                    'percentile_25': np.percentile(np_values, 25),
                    'percentile_75': np.percentile(np_values, 75),
                    'iqr': np.percentile(np_values, 75) - np.percentile(np_values, 25)
                })
            except Exception:
                pass
        
        return stats_dict

    def _calculate_data_quality_metrics(self) -> Dict[str, Any]:
        """Calculate metrics related to data quality and completeness."""
        total_entries = len(self.historical_data)
        
        if total_entries == 0:
            return {'no_data': True}
        
        # Check data completeness
        complete_entries = sum(1 for entry in self.historical_data 
                              if all(key in entry for key in ['timestamp', 'summary', 'module_coverage']))
        
        completeness_ratio = complete_entries / total_entries
        
        # Check temporal consistency
        timestamps = [datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00')) for entry in self.historical_data]
        time_gaps = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
        
        data_quality = {
            'total_entries': total_entries,
            'complete_entries': complete_entries,
            'completeness_ratio': completeness_ratio,
            'temporal_consistency': {
                'average_gap_hours': statistics.mean(time_gaps) / 3600 if time_gaps else 0,
                'max_gap_hours': max(time_gaps) / 3600 if time_gaps else 0,
                'min_gap_hours': min(time_gaps) / 3600 if time_gaps else 0
            },
            'data_freshness_hours': (datetime.now(timezone.utc) - max(timestamps)).total_seconds() / 3600 if timestamps else float('inf')
        }
        
        return data_quality

    def _calculate_trend_stability(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend stability metrics to assess prediction reliability."""
        if len(values) < 3:
            return {'insufficient_data': True}
        
        # Calculate moving averages to assess stability
        window_sizes = [3, 5, min(10, len(values))]
        stability_metrics = {}
        
        for window_size in window_sizes:
            if len(values) >= window_size:
                moving_avgs = []
                for i in range(len(values) - window_size + 1):
                    avg = sum(values[i:i + window_size]) / window_size
                    moving_avgs.append(avg)
                
                if len(moving_avgs) > 1:
                    avg_variance = statistics.variance(moving_avgs)
                    stability_metrics[f'moving_avg_{window_size}_variance'] = avg_variance
        
        # Calculate overall stability score
        if stability_metrics:
            avg_variance = statistics.mean(stability_metrics.values())
            stability_score = max(0, 1 - (avg_variance / 100))  # Normalize to 0-1 scale
            stability_metrics['overall_stability_score'] = stability_score
            
            if stability_score > 0.8:
                stability_metrics['stability_level'] = 'high'
            elif stability_score > 0.6:
                stability_metrics['stability_level'] = 'medium'
            else:
                stability_metrics['stability_level'] = 'low'
        
        return stability_metrics

    def _calculate_prediction_reliability(self) -> Dict[str, Any]:
        """Calculate metrics for assessing the reliability of trend predictions."""
        if len(self.historical_data) < 5:
            return {'insufficient_data': True}
        
        # Simple prediction reliability based on trend consistency
        recent_values = [entry['summary']['overall_coverage_percentage'] for entry in self.historical_data[-10:]]
        
        reliability_metrics = {
            'recent_data_points': len(recent_values),
            'recent_variance': statistics.variance(recent_values) if len(recent_values) > 1 else 0,
            'data_age_days': (datetime.now(timezone.utc) - 
                             datetime.fromisoformat(self.historical_data[-1]['timestamp'].replace('Z', '+00:00'))).days
        }
        
        # Calculate reliability score
        variance_penalty = min(reliability_metrics['recent_variance'] / 10, 0.5)  # Penalize high variance
        age_penalty = min(reliability_metrics['data_age_days'] / 7, 0.3)  # Penalize old data
        
        reliability_score = max(0, 1 - variance_penalty - age_penalty)
        reliability_metrics['reliability_score'] = reliability_score
        
        if reliability_score > 0.8:
            reliability_metrics['reliability_level'] = 'high'
        elif reliability_score > 0.6:
            reliability_metrics['reliability_level'] = 'medium'
        else:
            reliability_metrics['reliability_level'] = 'low'
        
        return reliability_metrics

    def _correlate_performance_data(self) -> None:
        """
        Correlate coverage trends with performance benchmark data per TST-PERF-001
        and TST-PERF-002 requirements for comprehensive quality metrics.
        """
        self.logger.debug("Correlating coverage trends with performance data...")
        
        performance_correlation = {
            'correlation_available': False,
            'data_loading_correlation': {},
            'transformation_correlation': {},
            'overall_performance_impact': {}
        }
        
        # Check for performance data availability
        performance_data_file = self.historical_data_dir / "performance_history.json"
        
        if performance_data_file.exists():
            try:
                with open(performance_data_file, 'r', encoding='utf-8') as f:
                    perf_data = json.load(f)
                
                performance_correlation = self._analyze_performance_correlation(perf_data)
                
            except Exception as e:
                self.logger.warning(f"Failed to load performance data: {e}")
                performance_correlation['error'] = str(e)
        else:
            self.logger.info("No performance data available for correlation analysis")
            performance_correlation['message'] = "Performance data not available - run benchmark tests to enable correlation analysis"
        
        self.trend_analysis['performance_correlation'] = performance_correlation

    def _analyze_performance_correlation(self, perf_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze correlation between coverage and performance metrics."""
        correlation_results = {
            'correlation_available': True,
            'data_loading_correlation': {'correlation_coefficient': 0.0, 'significance': 'not_calculated'},
            'transformation_correlation': {'correlation_coefficient': 0.0, 'significance': 'not_calculated'},
            'overall_performance_impact': {'impact_detected': False}
        }
        
        # Extract performance time series
        perf_entries = perf_data.get('entries', [])
        if not perf_entries:
            return correlation_results
        
        # Match performance data with coverage data by timestamp
        matched_data = []
        for cov_entry in self.historical_data:
            cov_timestamp = datetime.fromisoformat(cov_entry['timestamp'].replace('Z', '+00:00'))
            
            # Find closest performance entry (within 1 hour)
            closest_perf = None
            min_time_diff = timedelta(hours=1)
            
            for perf_entry in perf_entries:
                perf_timestamp = datetime.fromisoformat(perf_entry['timestamp'].replace('Z', '+00:00'))
                time_diff = abs(cov_timestamp - perf_timestamp)
                
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_perf = perf_entry
            
            if closest_perf:
                matched_data.append({
                    'coverage': cov_entry['summary']['overall_coverage_percentage'],
                    'data_loading_time': closest_perf.get('data_loading_time', 0),
                    'transformation_time': closest_perf.get('transformation_time', 0)
                })
        
        if len(matched_data) < 3:
            correlation_results['insufficient_matched_data'] = True
            return correlation_results
        
        # Calculate correlations if scipy is available
        if SCIPY_AVAILABLE:
            try:
                coverage_values = [d['coverage'] for d in matched_data]
                loading_times = [d['data_loading_time'] for d in matched_data]
                transform_times = [d['transformation_time'] for d in matched_data]
                
                # Correlation between coverage and data loading performance
                if any(t > 0 for t in loading_times):
                    loading_corr, loading_p = stats.pearsonr(coverage_values, loading_times)
                    correlation_results['data_loading_correlation'] = {
                        'correlation_coefficient': loading_corr,
                        'p_value': loading_p,
                        'significance': 'significant' if loading_p < 0.05 else 'not_significant',
                        'interpretation': self._interpret_correlation(loading_corr)
                    }
                
                # Correlation between coverage and transformation performance
                if any(t > 0 for t in transform_times):
                    transform_corr, transform_p = stats.pearsonr(coverage_values, transform_times)
                    correlation_results['transformation_correlation'] = {
                        'correlation_coefficient': transform_corr,
                        'p_value': transform_p,
                        'significance': 'significant' if transform_p < 0.05 else 'not_significant',
                        'interpretation': self._interpret_correlation(transform_corr)
                    }
                
                # Overall performance impact assessment
                significant_correlations = []
                if correlation_results['data_loading_correlation'].get('significance') == 'significant':
                    significant_correlations.append('data_loading')
                if correlation_results['transformation_correlation'].get('significance') == 'significant':
                    significant_correlations.append('transformation')
                
                correlation_results['overall_performance_impact'] = {
                    'impact_detected': len(significant_correlations) > 0,
                    'affected_areas': significant_correlations,
                    'matched_data_points': len(matched_data)
                }
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate performance correlations: {e}")
                correlation_results['correlation_error'] = str(e)
        
        return correlation_results

    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation coefficient magnitude and direction."""
        abs_corr = abs(correlation)
        direction = "positive" if correlation > 0 else "negative" if correlation < 0 else "none"
        
        if abs_corr >= 0.7:
            strength = "strong"
        elif abs_corr >= 0.5:
            strength = "moderate"
        elif abs_corr >= 0.3:
            strength = "weak"
        else:
            strength = "negligible"
        
        return f"{strength} {direction} correlation"

    def _generate_quality_predictions(self) -> None:
        """Generate quality predictions based on trend analysis and statistical models."""
        self.logger.debug("Generating quality predictions...")
        
        predictions = {
            'prediction_available': False,
            'short_term_prediction': {},
            'trend_based_forecast': {},
            'risk_assessment': {}
        }
        
        if len(self.historical_data) < 5:
            predictions['insufficient_data'] = True
            self.trend_analysis['quality_predictions'] = predictions
            return
        
        predictions['prediction_available'] = True
        
        # Short-term prediction (next data point)
        predictions['short_term_prediction'] = self._predict_short_term_coverage()
        
        # Trend-based forecast (next 7-30 days)
        predictions['trend_based_forecast'] = self._forecast_coverage_trend()
        
        # Risk assessment
        predictions['risk_assessment'] = self._assess_quality_risks()
        
        self.trend_analysis['quality_predictions'] = predictions

    def _predict_short_term_coverage(self) -> Dict[str, Any]:
        """Predict next coverage value based on recent trends."""
        recent_values = [entry['summary']['overall_coverage_percentage'] for entry in self.historical_data[-5:]]
        
        if len(recent_values) < 3:
            return {'insufficient_data': True}
        
        # Simple moving average prediction
        moving_avg = sum(recent_values) / len(recent_values)
        
        # Linear trend prediction if scipy available
        prediction_result = {
            'moving_average_prediction': moving_avg,
            'prediction_confidence': 'medium'
        }
        
        if SCIPY_AVAILABLE and len(recent_values) >= 3:
            try:
                x_values = list(range(len(recent_values)))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, recent_values)
                
                next_prediction = slope * len(recent_values) + intercept
                prediction_result.update({
                    'linear_trend_prediction': next_prediction,
                    'trend_slope': slope,
                    'prediction_r_squared': r_value ** 2,
                    'prediction_confidence': 'high' if abs(r_value) > 0.7 else 'medium'
                })
                
            except Exception:
                pass
        
        return prediction_result

    def _forecast_coverage_trend(self) -> Dict[str, Any]:
        """Forecast coverage trends for the next 7-30 days."""
        if len(self.historical_data) < 5:
            return {'insufficient_data': True}
        
        coverage_values = [entry['summary']['overall_coverage_percentage'] for entry in self.historical_data]
        
        forecast = {
            'forecast_horizon_days': 30,
            'trend_direction': 'stable',
            'confidence_level': 'low'
        }
        
        # Calculate recent trend direction
        recent_values = coverage_values[-10:]  # Last 10 data points
        if len(recent_values) >= 3:
            recent_trend = recent_values[-1] - recent_values[0]
            
            if recent_trend > 1:
                forecast['trend_direction'] = 'improving'
            elif recent_trend < -1:
                forecast['trend_direction'] = 'declining'
            else:
                forecast['trend_direction'] = 'stable'
            
            # Simple confidence assessment based on trend consistency
            direction_changes = 0
            for i in range(1, len(recent_values)):
                if (recent_values[i] - recent_values[i-1]) * recent_trend < 0:
                    direction_changes += 1
            
            consistency_ratio = 1 - (direction_changes / (len(recent_values) - 1))
            
            if consistency_ratio > 0.7:
                forecast['confidence_level'] = 'high'
            elif consistency_ratio > 0.5:
                forecast['confidence_level'] = 'medium'
            else:
                forecast['confidence_level'] = 'low'
            
            forecast['trend_consistency'] = consistency_ratio
        
        return forecast

    def _assess_quality_risks(self) -> Dict[str, Any]:
        """Assess quality risks based on trend analysis and threshold proximity."""
        risk_assessment = {
            'overall_risk_level': 'low',
            'risk_factors': [],
            'immediate_concerns': [],
            'recommendations': []
        }
        
        current_coverage = self.current_coverage['summary']['overall_coverage_percentage']
        threshold = self.quality_gates.get('coverage', {}).get('overall_coverage_threshold', 90.0)
        
        # Assess proximity to threshold
        threshold_gap = current_coverage - threshold
        
        if threshold_gap < 0:
            risk_assessment['risk_factors'].append({
                'factor': 'below_threshold',
                'severity': 'high',
                'description': f'Coverage {current_coverage:.1f}% is below threshold {threshold}%'
            })
            risk_assessment['overall_risk_level'] = 'high'
        elif threshold_gap < 2:
            risk_assessment['risk_factors'].append({
                'factor': 'near_threshold',
                'severity': 'medium',
                'description': f'Coverage {current_coverage:.1f}% is close to threshold {threshold}%'
            })
            risk_assessment['overall_risk_level'] = 'medium'
        
        # Assess trend direction risk
        overall_trends = self.trend_analysis.get('overall_trends', {})
        line_coverage_trend = overall_trends.get('line_coverage', {})
        
        if line_coverage_trend.get('trend_pattern', {}).get('pattern') == 'consistently_decreasing':
            risk_assessment['risk_factors'].append({
                'factor': 'declining_trend',
                'severity': 'medium',
                'description': 'Coverage shows consistently declining trend'
            })
            if risk_assessment['overall_risk_level'] == 'low':
                risk_assessment['overall_risk_level'] = 'medium'
        
        # Assess regression detection
        if self.trend_analysis.get('regression_analysis', {}).get('regression_detected'):
            regression_severity = self.trend_analysis['regression_analysis'].get('severity', 'low')
            risk_assessment['risk_factors'].append({
                'factor': 'regression_detected',
                'severity': regression_severity,
                'description': f'Coverage regression detected with {regression_severity} severity'
            })
            
            if regression_severity in ['high', 'critical']:
                risk_assessment['overall_risk_level'] = 'high'
            elif regression_severity == 'medium' and risk_assessment['overall_risk_level'] != 'high':
                risk_assessment['overall_risk_level'] = 'medium'
        
        # Generate recommendations based on risks
        risk_assessment['recommendations'] = self._generate_risk_recommendations(risk_assessment)
        
        return risk_assessment

    def _generate_risk_recommendations(self, risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on identified risks."""
        recommendations = []
        
        risk_level = risk_assessment['overall_risk_level']
        risk_factors = [rf['factor'] for rf in risk_assessment['risk_factors']]
        
        if 'below_threshold' in risk_factors:
            recommendations.extend([
                'Immediately review and improve test coverage for modules below threshold',
                'Focus on critical modules that require 100% coverage',
                'Run comprehensive test suite analysis to identify coverage gaps'
            ])
        
        if 'near_threshold' in risk_factors:
            recommendations.extend([
                'Proactively add tests for uncovered code paths',
                'Monitor coverage closely with more frequent measurements',
                'Review recent code changes that may have reduced coverage'
            ])
        
        if 'declining_trend' in risk_factors:
            recommendations.extend([
                'Investigate root causes of coverage decline',
                'Implement stricter code review processes',
                'Add pre-commit hooks for coverage validation'
            ])
        
        if 'regression_detected' in risk_factors:
            recommendations.extend([
                'Block merges until coverage regression is resolved',
                'Review recent commits that may have contributed to regression',
                'Run full test suite validation before deployment'
            ])
        
        # General recommendations based on risk level
        if risk_level == 'high':
            recommendations.extend([
                'Consider implementing emergency coverage improvement plan',
                'Increase test development velocity temporarily',
                'Review testing infrastructure for potential issues'
            ])
        
        return list(set(recommendations))  # Remove duplicates

    def _generate_trend_alerts(self) -> None:
        """Generate comprehensive alerts for detected trends and quality issues."""
        self.logger.debug("Generating trend alerts...")
        
        alerts = []
        
        # Regression alerts
        regression = self.trend_analysis.get('regression_analysis', {})
        if regression.get('regression_detected'):
            severity = regression.get('severity', 'medium')
            
            alert = {
                'type': 'regression',
                'severity': severity,
                'title': f'Coverage Regression Detected ({severity.title()} Severity)',
                'message': self._format_regression_alert_message(regression),
                'actionable': True,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'recommendations': []
            }
            
            # Add specific recommendations based on regression type
            if regression.get('overall_regression', {}).get('regression_detected'):
                alert['recommendations'].extend([
                    'Review recent code changes that may have reduced overall coverage',
                    'Run comprehensive test analysis to identify coverage gaps',
                    'Consider blocking merge until coverage is restored'
                ])
            
            if regression.get('critical_module_regression', {}).get('regression_detected'):
                alert['recommendations'].extend([
                    'Immediately address critical module coverage failures',
                    'Critical modules must maintain 100% coverage per TST-COV-002',
                    'Run focused testing on affected critical modules'
                ])
            
            alerts.append(alert)
        
        # Threshold violation alerts
        threshold_violations = regression.get('threshold_violations', [])
        for violation in threshold_violations:
            alert = {
                'type': 'threshold_violation',
                'severity': violation.get('severity', 'medium'),
                'title': f'Coverage Threshold Violation: {violation["type"]}',
                'message': self._format_threshold_violation_message(violation),
                'actionable': True,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'recommendations': [
                    f'Increase {violation["type"]} to meet {violation["threshold"]}% threshold',
                    'Review and improve test coverage in affected areas',
                    'Consider implementing additional test scenarios'
                ]
            }
            alerts.append(alert)
        
        # Performance correlation alerts
        perf_correlation = self.trend_analysis.get('performance_correlation', {})
        if perf_correlation.get('overall_performance_impact', {}).get('impact_detected'):
            alert = {
                'type': 'performance_correlation',
                'severity': 'medium',
                'title': 'Coverage-Performance Correlation Detected',
                'message': 'Significant correlation detected between coverage and performance metrics',
                'actionable': True,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'recommendations': [
                    'Review test execution efficiency and optimization opportunities',
                    'Monitor performance impact of coverage improvements',
                    'Consider balanced approach to coverage and performance'
                ]
            }
            alerts.append(alert)
        
        # Data quality alerts
        stats = self.trend_analysis.get('statistical_metrics', {})
        data_quality = stats.get('data_quality_metrics', {})
        
        if data_quality.get('completeness_ratio', 1.0) < 0.8:
            alert = {
                'type': 'data_quality',
                'severity': 'low',
                'title': 'Historical Data Quality Issues',
                'message': f'Data completeness ratio: {data_quality["completeness_ratio"]:.1%}',
                'actionable': True,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'recommendations': [
                    'Improve coverage data collection consistency',
                    'Validate data persistence mechanisms',
                    'Consider data cleaning and validation processes'
                ]
            }
            alerts.append(alert)
        
        # Prediction reliability alerts
        prediction_reliability = stats.get('prediction_reliability', {})
        if prediction_reliability.get('reliability_level') == 'low':
            alert = {
                'type': 'prediction_reliability',
                'severity': 'low',
                'title': 'Low Prediction Reliability',
                'message': 'Trend predictions have low reliability due to data variance or age',
                'actionable': True,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'recommendations': [
                    'Collect more consistent coverage data',
                    'Increase frequency of coverage measurements',
                    'Reduce variance in test execution environment'
                ]
            }
            alerts.append(alert)
        
        self.trend_analysis['alerts'] = alerts
        
        # Update analysis stats
        self.analysis_stats['warnings'].extend([alert for alert in alerts if alert['severity'] in ['low', 'medium']])
        self.analysis_stats['errors'].extend([alert for alert in alerts if alert['severity'] in ['high', 'critical']])

    def _format_regression_alert_message(self, regression: Dict[str, Any]) -> str:
        """Format detailed message for regression alerts."""
        messages = []
        
        overall_reg = regression.get('overall_regression', {})
        if overall_reg.get('regression_detected'):
            drop = overall_reg.get('percentage_drop', 0)
            messages.append(f"Overall coverage dropped by {drop:.1f}%")
        
        critical_reg = regression.get('critical_module_regression', {})
        if critical_reg.get('regression_detected'):
            affected_count = len(critical_reg.get('affected_modules', []))
            messages.append(f"{affected_count} critical module(s) below 100% coverage")
        
        threshold_violations = regression.get('threshold_violations', [])
        if threshold_violations:
            messages.append(f"{len(threshold_violations)} threshold violation(s) detected")
        
        return "; ".join(messages) if messages else "Coverage regression detected"

    def _format_threshold_violation_message(self, violation: Dict[str, Any]) -> str:
        """Format detailed message for threshold violation alerts."""
        if violation['type'] == 'critical_module_coverage':
            return f"Module {violation['module']} has {violation['actual']:.1f}% coverage (required: {violation['threshold']}%)"
        else:
            return f"{violation['type']} is {violation['actual']:.1f}% (required: {violation['threshold']}%)"

    def generate_trend_reports(self) -> None:
        """
        Generate comprehensive trend reports including HTML, JSON, and CSV formats
        with visualization charts and detailed analysis summaries.
        """
        self.logger.info("Generating comprehensive trend reports...")
        
        try:
            # Generate JSON report (machine-readable)
            self._generate_json_trend_report()
            
            # Generate CSV report (data analysis)
            self._generate_csv_trend_report()
            
            # Generate HTML report with visualizations
            self._generate_html_trend_report()
            
            # Generate text summary report
            self._generate_text_summary_report()
            
            self.logger.info("All trend reports generated successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to generate trend reports: {e}")
            raise

    def _generate_json_trend_report(self) -> None:
        """Generate comprehensive JSON trend report for programmatic analysis."""
        json_report = {
            'metadata': {
                'version': '2.0',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'generator': 'flyrigloader-coverage-trend-analyzer',
                'analysis_window_days': self.trend_window_days,
                'data_points_analyzed': len(self.historical_data)
            },
            'current_coverage': self.current_coverage,
            'trend_analysis': self.trend_analysis,
            'configuration': {
                'thresholds': self.thresholds,
                'quality_gates': self.quality_gates,
                'analysis_settings': {
                    'confidence_level': self.confidence_level,
                    'regression_sensitivity': self.regression_sensitivity,
                    'minimum_data_points': self.minimum_data_points
                }
            },
            'analysis_statistics': self.analysis_stats
        }
        
        output_file = self.output_dir / "coverage_trend_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Generated JSON trend report: {output_file}")

    def _generate_csv_trend_report(self) -> None:
        """Generate CSV trend report for data analysis and visualization tools."""
        import csv
        
        csv_file = self.output_dir / "coverage_trends.csv"
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            header = [
                'timestamp', 'overall_coverage', 'branch_coverage', 
                'total_statements', 'covered_statements', 'total_files',
                'critical_modules_passing', 'critical_modules_total',
                'build_number', 'branch_name', 'commit_sha'
            ]
            writer.writerow(header)
            
            # Write historical data
            for entry in self.historical_data:
                row = [
                    entry['timestamp'],
                    entry['summary']['overall_coverage_percentage'],
                    entry['summary']['branch_coverage_percentage'],
                    entry['summary']['total_statements'],
                    entry['summary']['covered_statements'],
                    entry['summary']['total_files'],
                    entry.get('critical_modules', {}).get('passing_critical', 0),
                    entry.get('critical_modules', {}).get('total_critical', 0),
                    entry.get('metadata', {}).get('build_number', 'unknown'),
                    entry.get('metadata', {}).get('branch_name', 'unknown'),
                    entry.get('metadata', {}).get('commit_sha', 'unknown')
                ]
                writer.writerow(row)
        
        self.logger.info(f"Generated CSV trend report: {csv_file}")

    def _generate_html_trend_report(self) -> None:
        """Generate HTML trend report with interactive visualizations."""
        html_report = self._create_html_trend_template()
        
        # Generate visualizations if matplotlib is available
        if MATPLOTLIB_AVAILABLE:
            try:
                self._generate_trend_visualizations()
                html_report = html_report.replace('{{CHARTS_AVAILABLE}}', 'true')
            except Exception as e:
                self.logger.warning(f"Failed to generate visualizations: {e}")
                html_report = html_report.replace('{{CHARTS_AVAILABLE}}', 'false')
        else:
            html_report = html_report.replace('{{CHARTS_AVAILABLE}}', 'false')
        
        # Replace template variables
        html_report = self._replace_html_template_variables(html_report)
        
        output_file = self.output_dir / "coverage_trends.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        self.logger.info(f"Generated HTML trend report: {output_file}")

    def _create_html_trend_template(self) -> str:
        """Create HTML template for trend report."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlyrigLoader Coverage Trend Analysis</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        .header {
            background: linear-gradient(135deg, #2E7D32, #4CAF50);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header .subtitle {
            margin-top: 10px;
            font-size: 1.1em;
            opacity: 0.9;
        }
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .card:hover {
            transform: translateY(-2px);
        }
        .card h3 {
            margin-top: 0;
            color: #2E7D32;
            border-bottom: 2px solid #E8F5E8;
            padding-bottom: 10px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 15px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        .metric-label {
            font-weight: 500;
        }
        .metric-value {
            font-weight: bold;
            font-size: 1.1em;
        }
        .status-excellent { color: #2E7D32; }
        .status-good { color: #FFA000; }
        .status-warning { color: #F57C00; }
        .status-critical { color: #D32F2F; }
        .alert {
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid;
        }
        .alert-critical {
            background: #ffebee;
            border-color: #f44336;
            color: #c62828;
        }
        .alert-high {
            background: #fff3e0;
            border-color: #ff9800;
            color: #e65100;
        }
        .alert-medium {
            background: #fff8e1;
            border-color: #ffc107;
            color: #f57c00;
        }
        .alert-low {
            background: #e8f5e8;
            border-color: #4caf50;
            color: #2e7d32;
        }
        .chart-container {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 20px 0;
            text-align: center;
        }
        .recommendations {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        .recommendations ul {
            padding-left: 20px;
        }
        .recommendations li {
            margin: 10px 0;
            line-height: 1.5;
        }
        .timestamp {
            text-align: center;
            color: #666;
            font-style: italic;
            margin-top: 30px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>FlyrigLoader Coverage Trend Analysis</h1>
        <div class="subtitle">Comprehensive automated coverage tracking and regression detection</div>
        <div class="subtitle">Generated: {{GENERATION_TIME}}</div>
    </div>

    <div class="dashboard">
        <div class="card">
            <h3>Current Coverage Status</h3>
            <div class="metric">
                <span class="metric-label">Overall Coverage:</span>
                <span class="metric-value status-{{OVERALL_STATUS}}">{{OVERALL_COVERAGE}}%</span>
            </div>
            <div class="metric">
                <span class="metric-label">Branch Coverage:</span>
                <span class="metric-value status-{{BRANCH_STATUS}}">{{BRANCH_COVERAGE}}%</span>
            </div>
            <div class="metric">
                <span class="metric-label">Total Files:</span>
                <span class="metric-value">{{TOTAL_FILES}}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Total Statements:</span>
                <span class="metric-value">{{TOTAL_STATEMENTS}}</span>
            </div>
        </div>

        <div class="card">
            <h3>Trend Analysis</h3>
            <div class="metric">
                <span class="metric-label">Trend Direction:</span>
                <span class="metric-value">{{TREND_DIRECTION}}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Data Points:</span>
                <span class="metric-value">{{DATA_POINTS}}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Analysis Window:</span>
                <span class="metric-value">{{ANALYSIS_WINDOW}} days</span>
            </div>
            <div class="metric">
                <span class="metric-label">Prediction Confidence:</span>
                <span class="metric-value">{{PREDICTION_CONFIDENCE}}</span>
            </div>
        </div>

        <div class="card">
            <h3>Quality Gates</h3>
            <div class="metric">
                <span class="metric-label">Overall Threshold:</span>
                <span class="metric-value status-{{OVERALL_GATE_STATUS}}">{{OVERALL_GATE_RESULT}}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Critical Modules:</span>
                <span class="metric-value status-{{CRITICAL_GATE_STATUS}}">{{CRITICAL_GATE_RESULT}}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Regression Detection:</span>
                <span class="metric-value status-{{REGRESSION_STATUS}}">{{REGRESSION_RESULT}}</span>
            </div>
        </div>

        <div class="card">
            <h3>Risk Assessment</h3>
            <div class="metric">
                <span class="metric-label">Overall Risk Level:</span>
                <span class="metric-value status-{{RISK_STATUS}}">{{RISK_LEVEL}}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Risk Factors:</span>
                <span class="metric-value">{{RISK_FACTORS_COUNT}}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Trend Stability:</span>
                <span class="metric-value">{{TREND_STABILITY}}</span>
            </div>
        </div>
    </div>

    {{CHARTS_SECTION}}

    <div class="card">
        <h3>Alerts and Notifications</h3>
        {{ALERTS_SECTION}}
    </div>

    <div class="recommendations">
        <h3>Recommendations</h3>
        {{RECOMMENDATIONS_SECTION}}
    </div>

    <div class="timestamp">
        Report generated by FlyrigLoader Coverage Trend Analyzer<br>
        Analysis completed: {{ANALYSIS_COMPLETION_TIME}}
    </div>
</body>
</html>
        """

    def _replace_html_template_variables(self, html_template: str) -> str:
        """Replace template variables with actual data values."""
        # Current coverage data
        current_summary = self.current_coverage['summary']
        overall_coverage = current_summary['overall_coverage_percentage']
        branch_coverage = current_summary['branch_coverage_percentage']
        
        # Determine status classes
        overall_status = self._get_status_class(overall_coverage, 90.0)
        branch_status = self._get_status_class(branch_coverage, 90.0)
        
        # Trend analysis data
        trend_analysis = self.trend_analysis
        overall_trends = trend_analysis.get('overall_trends', {})
        line_trend = overall_trends.get('line_coverage', {})
        trend_pattern = line_trend.get('trend_pattern', {})
        
        # Quality gates
        quality_gates = self.current_coverage.get('quality_gates', {})
        overall_gate = quality_gates.get('overall_threshold', {})
        critical_gate = quality_gates.get('critical_modules', {})
        
        # Risk assessment
        risk_assessment = trend_analysis.get('quality_predictions', {}).get('risk_assessment', {})
        
        # Regression analysis
        regression = trend_analysis.get('regression_analysis', {})
        
        # Replace variables
        replacements = {
            '{{GENERATION_TIME}}': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
            '{{OVERALL_COVERAGE}}': f"{overall_coverage:.1f}",
            '{{BRANCH_COVERAGE}}': f"{branch_coverage:.1f}",
            '{{TOTAL_FILES}}': str(current_summary['total_files']),
            '{{TOTAL_STATEMENTS}}': f"{current_summary['total_statements']:,}",
            '{{OVERALL_STATUS}}': overall_status,
            '{{BRANCH_STATUS}}': branch_status,
            '{{TREND_DIRECTION}}': trend_pattern.get('pattern', 'unknown').replace('_', ' ').title(),
            '{{DATA_POINTS}}': str(len(self.historical_data)),
            '{{ANALYSIS_WINDOW}}': str(self.trend_window_days),
            '{{PREDICTION_CONFIDENCE}}': trend_analysis.get('quality_predictions', {}).get('short_term_prediction', {}).get('prediction_confidence', 'unknown').title(),
            '{{OVERALL_GATE_STATUS}}': 'excellent' if overall_gate.get('passed', False) else 'critical',
            '{{OVERALL_GATE_RESULT}}': 'PASS' if overall_gate.get('passed', False) else 'FAIL',
            '{{CRITICAL_GATE_STATUS}}': 'excellent' if critical_gate.get('passed', False) else 'critical',
            '{{CRITICAL_GATE_RESULT}}': 'PASS' if critical_gate.get('passed', False) else 'FAIL',
            '{{REGRESSION_STATUS}}': 'critical' if regression.get('regression_detected', False) else 'excellent',
            '{{REGRESSION_RESULT}}': 'DETECTED' if regression.get('regression_detected', False) else 'NONE',
            '{{RISK_LEVEL}}': risk_assessment.get('overall_risk_level', 'unknown').title(),
            '{{RISK_STATUS}}': self._get_risk_status_class(risk_assessment.get('overall_risk_level', 'low')),
            '{{RISK_FACTORS_COUNT}}': str(len(risk_assessment.get('risk_factors', []))),
            '{{TREND_STABILITY}}': trend_analysis.get('statistical_metrics', {}).get('trend_stability', {}).get('stability_level', 'unknown').title(),
            '{{ANALYSIS_COMPLETION_TIME}}': self.analysis_stats['start_time'].strftime('%Y-%m-%d %H:%M:%S UTC')
        }
        
        # Charts section
        charts_section = self._generate_charts_section()
        replacements['{{CHARTS_SECTION}}'] = charts_section
        
        # Alerts section
        alerts_section = self._generate_alerts_section()
        replacements['{{ALERTS_SECTION}}'] = alerts_section
        
        # Recommendations section
        recommendations_section = self._generate_recommendations_section()
        replacements['{{RECOMMENDATIONS_SECTION}}'] = recommendations_section
        
        # Apply replacements
        for placeholder, value in replacements.items():
            html_template = html_template.replace(placeholder, str(value))
        
        return html_template

    def _get_status_class(self, value: float, threshold: float) -> str:
        """Get CSS status class based on value and threshold."""
        if value >= threshold:
            return 'excellent'
        elif value >= threshold - 5:
            return 'good'
        elif value >= threshold - 10:
            return 'warning'
        else:
            return 'critical'

    def _get_risk_status_class(self, risk_level: str) -> str:
        """Get CSS status class for risk level."""
        risk_classes = {
            'low': 'excellent',
            'medium': 'warning',
            'high': 'critical'
        }
        return risk_classes.get(risk_level, 'warning')

    def _generate_charts_section(self) -> str:
        """Generate charts section for HTML report."""
        if not MATPLOTLIB_AVAILABLE:
            return '''
            <div class="chart-container">
                <h3>Coverage Trend Visualization</h3>
                <p><em>Visualization charts not available (matplotlib not installed)</em></p>
                <p>Install matplotlib to enable trend visualization: <code>pip install matplotlib</code></p>
            </div>
            '''
        
        return '''
        <div class="chart-container">
            <h3>Coverage Trend Visualization</h3>
            <img src="coverage_trend_chart.png" alt="Coverage Trend Chart" style="max-width: 100%; height: auto;">
        </div>
        <div class="chart-container">
            <h3>Module Coverage Heatmap</h3>
            <img src="module_coverage_heatmap.png" alt="Module Coverage Heatmap" style="max-width: 100%; height: auto;">
        </div>
        '''

    def _generate_alerts_section(self) -> str:
        """Generate alerts section for HTML report."""
        alerts = self.trend_analysis.get('alerts', [])
        
        if not alerts:
            return '<p><em>No alerts detected. Coverage trends are within acceptable parameters.</em></p>'
        
        alerts_html = []
        for alert in alerts:
            severity_class = f"alert-{alert.get('severity', 'medium')}"
            alerts_html.append(f'''
            <div class="alert {severity_class}">
                <strong>{alert.get('title', 'Alert')}</strong><br>
                {alert.get('message', 'No message available')}
            </div>
            ''')
        
        return ''.join(alerts_html)

    def _generate_recommendations_section(self) -> str:
        """Generate recommendations section for HTML report."""
        recommendations = []
        
        # Get recommendations from risk assessment
        risk_assessment = self.trend_analysis.get('quality_predictions', {}).get('risk_assessment', {})
        recommendations.extend(risk_assessment.get('recommendations', []))
        
        # Get recommendations from alerts
        alerts = self.trend_analysis.get('alerts', [])
        for alert in alerts:
            recommendations.extend(alert.get('recommendations', []))
        
        # Remove duplicates and sort
        unique_recommendations = list(set(recommendations))
        
        if not unique_recommendations:
            return '<p><em>No specific recommendations at this time. Continue monitoring coverage trends.</em></p>'
        
        recommendations_html = ['<ul>']
        for rec in unique_recommendations:
            recommendations_html.append(f'<li>{rec}</li>')
        recommendations_html.append('</ul>')
        
        return ''.join(recommendations_html)

    def _generate_trend_visualizations(self) -> None:
        """Generate trend visualization charts using matplotlib."""
        if not MATPLOTLIB_AVAILABLE or len(self.historical_data) < 2:
            return
        
        try:
            # Set up matplotlib for better-looking charts
            plt.style.use('default')
            
            # Generate coverage trend chart
            self._generate_coverage_trend_chart()
            
            # Generate module coverage heatmap
            self._generate_module_coverage_heatmap()
            
        except Exception as e:
            self.logger.warning(f"Failed to generate visualizations: {e}")

    def _generate_coverage_trend_chart(self) -> None:
        """Generate line chart showing coverage trends over time."""
        # Extract time series data
        timestamps = [datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00')) for entry in self.historical_data]
        overall_coverage = [entry['summary']['overall_coverage_percentage'] for entry in self.historical_data]
        branch_coverage = [entry['summary']['branch_coverage_percentage'] for entry in self.historical_data]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot coverage trends
        ax.plot(timestamps, overall_coverage, 'o-', label='Overall Coverage', linewidth=2, markersize=4)
        ax.plot(timestamps, branch_coverage, 's-', label='Branch Coverage', linewidth=2, markersize=4)
        
        # Add threshold lines
        overall_threshold = self.quality_gates.get('coverage', {}).get('overall_coverage_threshold', 90.0)
        ax.axhline(y=overall_threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({overall_threshold}%)')
        
        # Formatting
        ax.set_xlabel('Date')
        ax.set_ylabel('Coverage Percentage (%)')
        ax.set_title('FlyrigLoader Coverage Trends Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(timestamps) // 10)))
        plt.xticks(rotation=45)
        
        # Adjust layout and save
        plt.tight_layout()
        chart_path = self.output_dir / "coverage_trend_chart.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Generated coverage trend chart: {chart_path}")

    def _generate_module_coverage_heatmap(self) -> None:
        """Generate heatmap showing module coverage status."""
        if not self.current_coverage.get('module_coverage'):
            return
        
        # Prepare data for heatmap
        modules = list(self.current_coverage['module_coverage'].keys())
        coverages = [self.current_coverage['module_coverage'][mod]['line_coverage'] for mod in modules]
        
        # Limit to top modules if too many
        if len(modules) > 20:
            # Sort by coverage and take worst performing modules
            module_coverage_pairs = list(zip(modules, coverages))
            module_coverage_pairs.sort(key=lambda x: x[1])
            modules = [pair[0] for pair in module_coverage_pairs[:20]]
            coverages = [pair[1] for pair in module_coverage_pairs[:20]]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, max(6, len(modules) * 0.3)))
        
        # Create color map
        colors = []
        for coverage in coverages:
            if coverage >= 95:
                colors.append('#2E7D32')  # Green
            elif coverage >= 90:
                colors.append('#FFA000')  # Orange
            elif coverage >= 80:
                colors.append('#F57C00')  # Deep Orange
            else:
                colors.append('#D32F2F')  # Red
        
        # Create horizontal bar chart
        y_pos = range(len(modules))
        bars = ax.barh(y_pos, coverages, color=colors)
        
        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels([mod.split('/')[-1] for mod in modules])  # Show only filename
        ax.set_xlabel('Coverage Percentage (%)')
        ax.set_title('Module Coverage Status')
        ax.set_xlim(0, 100)
        
        # Add percentage labels on bars
        for i, (bar, coverage) in enumerate(zip(bars, coverages)):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                   f'{coverage:.1f}%', ha='left', va='center', fontsize=8)
        
        # Add threshold line
        threshold = 90.0
        ax.axvline(x=threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({threshold}%)')
        ax.legend()
        
        plt.tight_layout()
        chart_path = self.output_dir / "module_coverage_heatmap.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Generated module coverage heatmap: {chart_path}")

    def _generate_text_summary_report(self) -> None:
        """Generate text summary report for console output and logging."""
        summary_lines = [
            "="*80,
            "FLYRIGLOADER COVERAGE TREND ANALYSIS SUMMARY",
            "="*80,
            "",
            f"Analysis Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Analysis Window: {self.trend_window_days} days",
            f"Historical Data Points: {len(self.historical_data)}",
            "",
            "CURRENT COVERAGE STATUS:",
            f"  Overall Coverage: {self.current_coverage['summary']['overall_coverage_percentage']:.2f}%",
            f"  Branch Coverage: {self.current_coverage['summary']['branch_coverage_percentage']:.2f}%",
            f"  Total Files: {self.current_coverage['summary']['total_files']}",
            f"  Total Statements: {self.current_coverage['summary']['total_statements']:,}",
            ""
        ]
        
        # Quality gates status
        quality_gates = self.current_coverage.get('quality_gates', {})
        summary_lines.extend([
            "QUALITY GATES STATUS:",
            f"  Overall Threshold: {'PASS' if quality_gates.get('overall_threshold', {}).get('passed', False) else 'FAIL'}",
            f"  Critical Modules: {'PASS' if quality_gates.get('critical_modules', {}).get('passed', False) else 'FAIL'}",
            ""
        ])
        
        # Regression analysis
        regression = self.trend_analysis.get('regression_analysis', {})
        summary_lines.extend([
            "REGRESSION ANALYSIS:",
            f"  Regression Detected: {'YES' if regression.get('regression_detected', False) else 'NO'}",
            f"  Severity: {regression.get('severity', 'none').title()}",
            f"  Confidence: {regression.get('confidence', 'low').title()}",
            ""
        ])
        
        # Alerts
        alerts = self.trend_analysis.get('alerts', [])
        if alerts:
            summary_lines.extend(["ALERTS:"])
            for alert in alerts[:5]:  # Show top 5 alerts
                summary_lines.append(f"  [{alert.get('severity', 'medium').upper()}] {alert.get('title', 'Alert')}")
            if len(alerts) > 5:
                summary_lines.append(f"  ... and {len(alerts) - 5} more alerts")
            summary_lines.append("")
        
        # Risk assessment
        risk_assessment = self.trend_analysis.get('quality_predictions', {}).get('risk_assessment', {})
        summary_lines.extend([
            "RISK ASSESSMENT:",
            f"  Overall Risk Level: {risk_assessment.get('overall_risk_level', 'unknown').title()}",
            f"  Risk Factors: {len(risk_assessment.get('risk_factors', []))}",
            ""
        ])
        
        # Recommendations
        recommendations = risk_assessment.get('recommendations', [])
        if recommendations:
            summary_lines.extend(["TOP RECOMMENDATIONS:"])
            for rec in recommendations[:3]:  # Show top 3 recommendations
                summary_lines.append(f"  • {rec}")
            summary_lines.append("")
        
        summary_lines.extend([
            "="*80,
            f"Analysis completed in {time.time() - self.start_time:.2f} seconds",
            "="*80
        ])
        
        # Write to file
        summary_text = "\n".join(summary_lines)
        output_file = self.output_dir / "coverage_trend_summary.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        # Also log to console
        self.logger.info("Coverage Trend Analysis Summary:")
        for line in summary_lines[2:-2]:  # Skip decorative borders
            if line.strip():
                self.logger.info(line)
        
        self.logger.info(f"Generated text summary report: {output_file}")

    def validate_quality_gates(self) -> bool:
        """
        Validate current coverage against quality gates and return pass/fail status.
        Implements TST-COV-004 requirement to block merges when coverage drops below thresholds.
        
        Returns:
            True if all quality gates pass, False otherwise
        """
        self.logger.info("Validating quality gates...")
        
        quality_gates_passed = True
        
        # Check overall coverage threshold
        overall_threshold = self.quality_gates.get('coverage', {}).get('overall_coverage_threshold', 90.0)
        current_overall = self.current_coverage['summary']['overall_coverage_percentage']
        
        if current_overall < overall_threshold:
            quality_gates_passed = False
            self.logger.error(f"Overall coverage {current_overall:.2f}% below threshold {overall_threshold}%")
        else:
            self.logger.info(f"Overall coverage {current_overall:.2f}% meets threshold {overall_threshold}%")
        
        # Check branch coverage threshold
        branch_threshold = self.quality_gates.get('coverage', {}).get('overall_branch_threshold', 90.0)
        current_branch = self.current_coverage['summary']['branch_coverage_percentage']
        
        if current_branch < branch_threshold:
            quality_gates_passed = False
            self.logger.error(f"Branch coverage {current_branch:.2f}% below threshold {branch_threshold}%")
        else:
            self.logger.info(f"Branch coverage {current_branch:.2f}% meets threshold {branch_threshold}%")
        
        # Check critical module coverage
        critical_threshold = self.quality_gates.get('coverage', {}).get('critical_module_coverage_threshold', 100.0)
        critical_failures = []
        
        for module_path, module_data in self.current_coverage.get('module_coverage', {}).items():
            if self._is_critical_module(module_path):
                module_coverage = module_data['line_coverage']
                if module_coverage < critical_threshold:
                    critical_failures.append((module_path, module_coverage))
        
        if critical_failures:
            quality_gates_passed = False
            self.logger.error(f"Critical module coverage failures: {len(critical_failures)}")
            for module_path, coverage in critical_failures:
                self.logger.error(f"  {module_path}: {coverage:.2f}% < {critical_threshold}%")
        else:
            self.logger.info("All critical modules meet coverage requirements")
        
        # Check for regression
        if self.trend_analysis.get('regression_analysis', {}).get('regression_detected', False):
            regression_severity = self.trend_analysis['regression_analysis'].get('severity', 'low')
            if regression_severity in ['high', 'critical']:
                quality_gates_passed = False
                self.logger.error(f"Coverage regression detected with {regression_severity} severity")
        
        # Update analysis stats
        self.analysis_stats['quality_gates_passed'] = quality_gates_passed
        
        if quality_gates_passed:
            self.logger.info("All quality gates PASSED")
        else:
            self.logger.error("Quality gates FAILED - merge should be blocked per TST-COV-004")
        
        return quality_gates_passed

    def run_complete_analysis(self) -> int:
        """
        Execute the complete coverage trend analysis pipeline including data collection,
        trend analysis, regression detection, and report generation.
        
        Returns:
            Exit code: 0 for success, 1 for quality gate failures, 2 for analysis errors
        """
        self.logger.info("Starting complete coverage trend analysis pipeline...")
        
        try:
            # Load configuration
            self.load_configuration()
            
            # Collect current coverage data
            self.collect_current_coverage_data()
            
            # Load historical data
            self.load_historical_data()
            
            # Persist current data for future analysis
            self.persist_current_data()
            
            # Perform comprehensive trend analysis
            self.perform_trend_analysis()
            
            # Generate all reports
            self.generate_trend_reports()
            
            # Validate quality gates
            quality_gates_passed = self.validate_quality_gates()
            
            # Update final analysis stats
            self.analysis_stats['analysis_completed'] = True
            self.analysis_stats['end_time'] = datetime.now(timezone.utc)
            self.analysis_stats['duration'] = time.time() - self.start_time
            
            # Log completion summary
            self.logger.info("Coverage trend analysis completed successfully")
            self.logger.info(f"Analysis duration: {self.analysis_stats['duration']:.2f} seconds")
            self.logger.info(f"Historical data points: {len(self.historical_data)}")
            self.logger.info(f"Alerts generated: {len(self.trend_analysis.get('alerts', []))}")
            self.logger.info(f"Quality gates: {'PASSED' if quality_gates_passed else 'FAILED'}")
            
            # Return appropriate exit code
            if not quality_gates_passed:
                self.logger.error("Returning exit code 1 due to quality gate failures")
                return 1
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Coverage trend analysis failed: {e}")
            self.analysis_stats['errors'].append(str(e))
            return 2


def main() -> int:
    """
    Main entry point for the coverage trend analysis script.
    
    Supports command-line arguments for configuration and provides
    comprehensive error handling and CI/CD integration.
    
    Returns:
        Exit code: 0 for success, 1 for quality gate failures, 2 for analysis errors
    """
    parser = argparse.ArgumentParser(
        description="FlyrigLoader Coverage Trend Analyzer - Automated coverage tracking with regression detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run complete analysis with default settings
  %(prog)s --verbose                          # Enable verbose logging
  %(prog)s --coverage-data .coverage          # Specify coverage data file
  %(prog)s --historical-dir ./trends          # Specify historical data directory
  %(prog)s --output-dir ./reports             # Specify output directory
  %(prog)s --thresholds ./custom-thresholds.json  # Use custom thresholds

Analysis Features:
  - Historical coverage data persistence and trend tracking
  - Statistical regression detection with configurable sensitivity
  - Performance benchmark correlation analysis
  - Automated alerting for coverage degradation
  - Confidence interval analysis and significance testing
  - CI/CD integration with quality gate enforcement

Quality Gates (TST-COV-004):
  The script validates coverage against configurable thresholds and returns
  appropriate exit codes for CI/CD integration:
  - Exit code 0: All quality gates passed
  - Exit code 1: Quality gate failures (blocks merge)
  - Exit code 2: Analysis errors or system failures

Requirements Implementation:
  - Section 0.2.5: Coverage trend tracking over time
  - TST-COV-004: Block merges when coverage drops below thresholds
  - TST-PERF-001: Data loading SLA validation correlation
  - Section 3.6.4: Quality metrics dashboard integration
        """
    )
    
    parser.add_argument(
        '--coverage-data',
        help='Path to coverage data file (default: .coverage)',
        default=None
    )
    
    parser.add_argument(
        '--thresholds',
        help='Path to coverage thresholds JSON file (default: tests/coverage/coverage-thresholds.json)',
        default=None
    )
    
    parser.add_argument(
        '--quality-gates',
        help='Path to quality gates YAML file (default: tests/coverage/quality-gates.yml)',
        default=None
    )
    
    parser.add_argument(
        '--historical-dir',
        help='Directory for historical trend data (default: tests/coverage/historical)',
        default=None
    )
    
    parser.add_argument(
        '--output-dir',
        help='Output directory for trend reports (default: tests/coverage/trends)',
        default=None
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging output'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='FlyrigLoader Coverage Trend Analyzer 1.0.0'
    )

    args = parser.parse_args()

    # Initialize and run the coverage trend analyzer
    try:
        analyzer = CoverageTrendAnalyzer(
            coverage_data_file=args.coverage_data,
            thresholds_file=args.thresholds,
            quality_gates_file=args.quality_gates,
            historical_data_dir=args.historical_dir,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
        
        return analyzer.run_complete_analysis()
        
    except KeyboardInterrupt:
        print("\nCoverage trend analysis interrupted by user")
        return 2
    except Exception as e:
        print(f"Fatal error: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())