#!/usr/bin/env python3
"""
FlyrigLoader Coverage Trend Analysis Engine

Comprehensive statistical analysis, regression detection, and historical tracking system implementing
coverage trend monitoring to enforce the 90% coverage threshold and support quality gate validation
essential for research-grade software reliability per Section 6.6.2.5 of the enhanced testing strategy.

This script serves as the central coverage trend analysis orchestrator for the flyrigloader Coverage
Enhancement System, providing advanced statistical analysis with confidence intervals, regression
detection, and comprehensive historical tracking capabilities supporting CI/CD integration and
quality gate enforcement.

Key Features:
- Historical coverage trend monitoring with 30-day artifact retention
- Statistical analysis with confidence intervals for regression detection
- Integration with quality gates for coverage degradation alerts
- Comprehensive trend visualization and reporting
- CI/CD pipeline integration with automated artifact collection
- Support for enhanced testing strategy compliance validation

Author: Blitzy agent
Version: 1.0.0
Requirements: Section 6.6.2.5, Section 8.4, Section 8.5, Section 8.6
"""

import argparse
import json
import os
import statistics
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Standard library imports for statistical analysis and data processing
import math
import subprocess
from dataclasses import asdict, dataclass, field

# Try to import third-party libraries with graceful fallbacks
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("ℹ️  NumPy not available, using built-in statistics")

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Install with: pip install pyyaml>=6.0")
    sys.exit(1)


class CoverageTrendException(Exception):
    """Base exception for coverage trend analysis failures."""
    pass


class HistoricalDataError(CoverageTrendException):
    """Raised when historical data operations fail."""
    pass


class StatisticalAnalysisError(CoverageTrendException):
    """Raised when statistical analysis fails."""
    pass


class RegressionDetectionError(CoverageTrendException):
    """Raised when regression detection fails."""
    pass


class QualityGateAlertError(CoverageTrendException):
    """Raised when quality gate alert processing fails."""
    pass


@dataclass
class CoverageMeasurement:
    """Individual coverage measurement data point."""
    timestamp: str
    overall_coverage: float
    line_coverage: float
    branch_coverage: float
    function_coverage: float
    build_number: str = "unknown"
    commit_hash: str = "unknown"
    branch_name: str = "unknown"
    environment: str = "unknown"
    total_lines: int = 0
    covered_lines: int = 0
    total_branches: int = 0
    covered_branches: int = 0
    critical_modules_coverage: float = 0.0
    quality_gates_passed: int = 0
    quality_gates_total: int = 0


@dataclass
class TrendStatistics:
    """Statistical analysis results for coverage trends."""
    mean: float
    median: float
    std_deviation: float
    variance: float
    min_value: float
    max_value: float
    confidence_interval_95: Tuple[float, float]
    trend_slope: float
    trend_direction: str  # "increasing", "decreasing", "stable"
    r_squared: float
    data_points: int
    period_days: int


@dataclass
class RegressionAnalysis:
    """Regression detection analysis results."""
    regression_detected: bool
    severity: str  # "none", "minor", "moderate", "severe", "critical"
    coverage_drop: float
    confidence_level: float
    recent_measurements: List[float]
    baseline_average: float
    affected_areas: List[str] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class QualityGateAlert:
    """Quality gate alert information."""
    alert_type: str  # "coverage_degradation", "threshold_violation", "regression_detected"
    severity: str  # "info", "warning", "error", "critical"
    message: str
    timestamp: str
    coverage_value: float
    threshold_value: float
    details: Dict[str, Any] = field(default_factory=dict)


class CoverageTrendAnalyzer:
    """
    Comprehensive coverage trend analysis engine implementing statistical analysis,
    regression detection, and historical tracking per Section 6.6.2.5 requirements.
    
    This class orchestrates coverage trend monitoring with advanced statistical analysis,
    confidence interval calculations, and regression detection to support the enhanced
    testing strategy's 90% line coverage threshold enforcement and quality gate validation.
    """

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize coverage trend analyzer with configuration loading.
        
        Args:
            project_root: Root directory of the project (defaults to current working directory)
        """
        self.project_root = Path(project_root or os.getcwd()).resolve()
        self.coverage_dir = self.project_root / "scripts" / "coverage"
        self.historical_dir = self.coverage_dir / "historical"
        self.reports_dir = self.coverage_dir / "reports"
        self.artifacts_dir = self.coverage_dir / "artifacts"
        
        # Ensure directories exist
        self.historical_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration files
        self.history_file = self.historical_dir / "coverage_history.json"
        self.thresholds_file = self.coverage_dir / "coverage-thresholds.json"
        self.quality_gates_file = self.coverage_dir / "quality-gates.yml"
        
        # Analysis state
        self.historical_data: List[CoverageMeasurement] = []
        self.current_measurement: Optional[CoverageMeasurement] = None
        self.trend_statistics: Optional[TrendStatistics] = None
        self.regression_analysis: Optional[RegressionAnalysis] = None
        self.quality_alerts: List[QualityGateAlert] = []
        
        # Configuration
        self.coverage_thresholds = self._load_coverage_thresholds()
        self.quality_gates_config = self._load_quality_gates_config()
        
        print(f"✅ Coverage trend analyzer initialized")
        print(f"   📁 Historical data: {self.history_file}")
        print(f"   📊 Reports output: {self.reports_dir}")
        print(f"   🔧 Configuration loaded from scripts/coverage/")

    def _load_coverage_thresholds(self) -> Dict[str, Any]:
        """Load coverage thresholds configuration from scripts/coverage/."""
        try:
            if self.thresholds_file.exists():
                with open(self.thresholds_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Fallback default configuration
                return {
                    "global_settings": {
                        "overall_threshold": {
                            "line_coverage": 90.0,
                            "branch_coverage": 85.0,
                            "function_coverage": 90.0
                        },
                        "regression_detection": {
                            "sensitivity": "high",
                            "min_drop_threshold": 2.0,
                            "confidence_level": 0.95
                        }
                    }
                }
        except Exception as e:
            print(f"⚠️  Warning: Failed to load coverage thresholds: {e}")
            # Return minimal default configuration
            return {
                "global_settings": {
                    "overall_threshold": {"line_coverage": 90.0},
                    "regression_detection": {"min_drop_threshold": 2.0}
                }
            }

    def _load_quality_gates_config(self) -> Dict[str, Any]:
        """Load quality gates configuration from scripts/coverage/."""
        try:
            if self.quality_gates_file.exists():
                with open(self.quality_gates_file, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                # Fallback default configuration
                return {
                    "quality_gates": {
                        "coverage_threshold": {
                            "enabled": True,
                            "line_coverage_min": 90.0,
                            "branch_coverage_min": 85.0
                        },
                        "regression_detection": {
                            "enabled": True,
                            "sensitivity": "high"
                        },
                        "alert_configuration": {
                            "coverage_degradation": "warning",
                            "threshold_violation": "error",
                            "regression_detected": "critical"
                        }
                    }
                }
        except Exception as e:
            print(f"⚠️  Warning: Failed to load quality gates configuration: {e}")
            # Return minimal default configuration
            return {
                "quality_gates": {
                    "coverage_threshold": {"enabled": True, "line_coverage_min": 90.0},
                    "regression_detection": {"enabled": True}
                }
            }

    def load_historical_data(self) -> None:
        """Load historical coverage data from persistent storage."""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert dictionaries back to CoverageMeasurement objects
                self.historical_data = []
                for item in data.get('measurements', []):
                    measurement = CoverageMeasurement(**item)
                    self.historical_data.append(measurement)
                
                # Sort by timestamp to ensure chronological order
                self.historical_data.sort(key=lambda x: x.timestamp)
                
                print(f"✅ Loaded {len(self.historical_data)} historical measurements")
                
                # Apply 30-day retention policy
                self._apply_retention_policy()
                
            else:
                print("ℹ️  No historical data file found, starting fresh")
                self.historical_data = []
                
        except Exception as e:
            raise HistoricalDataError(f"Failed to load historical data: {e}")

    def _apply_retention_policy(self) -> None:
        """Apply 30-day retention policy for historical data per Section 8.6."""
        try:
            if not self.historical_data:
                return
            
            # Calculate cutoff date (30 days ago)
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
            
            # Filter measurements within retention period
            original_count = len(self.historical_data)
            self.historical_data = [
                measurement for measurement in self.historical_data
                if datetime.fromisoformat(measurement.timestamp.replace('Z', '+00:00')) >= cutoff_date
            ]
            
            removed_count = original_count - len(self.historical_data)
            if removed_count > 0:
                print(f"🗑️  Applied 30-day retention policy: removed {removed_count} old measurements")
                
        except Exception as e:
            print(f"⚠️  Warning: Failed to apply retention policy: {e}")

    def add_current_measurement(self, coverage_data: Dict[str, Any]) -> None:
        """Add current coverage measurement to historical data."""
        try:
            # Extract coverage metrics from provided data
            current_time = datetime.now(timezone.utc).isoformat()
            
            # Handle different input formats from generate-coverage-reports.py
            if 'coverage_data' in coverage_data:
                overall_data = coverage_data['coverage_data'].get('overall', {})
                meta_data = coverage_data.get('meta', {})
                build_data = coverage_data.get('build_info', {})
            else:
                # Direct format
                overall_data = coverage_data
                meta_data = {}
                build_data = {}
            
            # Create measurement object
            self.current_measurement = CoverageMeasurement(
                timestamp=current_time,
                overall_coverage=overall_data.get('overall_coverage_percentage', 0.0),
                line_coverage=overall_data.get('line_coverage_percentage', 0.0),
                branch_coverage=overall_data.get('branch_coverage_percentage', 0.0),
                function_coverage=overall_data.get('function_coverage_percentage', 0.0),
                build_number=build_data.get('build_number', os.environ.get('GITHUB_RUN_NUMBER', 'local')),
                commit_hash=build_data.get('commit_hash', os.environ.get('GITHUB_SHA', 'unknown')),
                branch_name=build_data.get('branch_name', os.environ.get('GITHUB_REF_NAME', 'unknown')),
                environment=build_data.get('environment', os.environ.get('CI_ENVIRONMENT', 'local')),
                total_lines=overall_data.get('total_lines', 0),
                covered_lines=overall_data.get('covered_lines', 0),
                total_branches=overall_data.get('total_branches', 0),
                covered_branches=overall_data.get('covered_branches', 0),
                critical_modules_coverage=coverage_data.get('critical_modules_coverage', 0.0),
                quality_gates_passed=sum(1 for v in coverage_data.get('quality_gates', {}).values() if v),
                quality_gates_total=len(coverage_data.get('quality_gates', {}))
            )
            
            # Add to historical data
            self.historical_data.append(self.current_measurement)
            
            print(f"✅ Added current measurement: {self.current_measurement.overall_coverage:.2f}% coverage")
            
        except Exception as e:
            raise HistoricalDataError(f"Failed to add current measurement: {e}")

    def calculate_trend_statistics(self, days: int = 30) -> TrendStatistics:
        """Calculate comprehensive statistical analysis for coverage trends."""
        try:
            if len(self.historical_data) < 2:
                print("⚠️  Insufficient data for trend analysis (need at least 2 measurements)")
                return TrendStatistics(
                    mean=0.0, median=0.0, std_deviation=0.0, variance=0.0,
                    min_value=0.0, max_value=0.0, confidence_interval_95=(0.0, 0.0),
                    trend_slope=0.0, trend_direction="unknown", r_squared=0.0,
                    data_points=len(self.historical_data), period_days=days
                )
            
            # Filter data within specified time period
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            recent_data = [
                measurement for measurement in self.historical_data
                if datetime.fromisoformat(measurement.timestamp.replace('Z', '+00:00')) >= cutoff_date
            ]
            
            if len(recent_data) < 2:
                print(f"⚠️  Insufficient recent data for {days}-day trend analysis")
                recent_data = self.historical_data[-min(len(self.historical_data), 10):]
            
            # Extract coverage values
            coverage_values = [measurement.overall_coverage for measurement in recent_data]
            
            # Calculate basic statistics
            mean_coverage = statistics.mean(coverage_values)
            median_coverage = statistics.median(coverage_values)
            
            if len(coverage_values) > 1:
                std_dev = statistics.stdev(coverage_values)
                variance = statistics.variance(coverage_values)
            else:
                std_dev = 0.0
                variance = 0.0
            
            min_coverage = min(coverage_values)
            max_coverage = max(coverage_values)
            
            # Calculate confidence interval (95%)
            if len(coverage_values) > 1 and std_dev > 0:
                margin_of_error = 1.96 * (std_dev / math.sqrt(len(coverage_values)))
                confidence_interval = (
                    max(0.0, mean_coverage - margin_of_error),
                    min(100.0, mean_coverage + margin_of_error)
                )
            else:
                confidence_interval = (mean_coverage, mean_coverage)
            
            # Calculate trend slope and direction using linear regression
            trend_slope, r_squared, trend_direction = self._calculate_trend_slope(recent_data)
            
            self.trend_statistics = TrendStatistics(
                mean=mean_coverage,
                median=median_coverage,
                std_deviation=std_dev,
                variance=variance,
                min_value=min_coverage,
                max_value=max_coverage,
                confidence_interval_95=confidence_interval,
                trend_slope=trend_slope,
                trend_direction=trend_direction,
                r_squared=r_squared,
                data_points=len(recent_data),
                period_days=min(days, len(recent_data))
            )
            
            print(f"📊 Calculated trend statistics:")
            print(f"   📈 Mean coverage: {mean_coverage:.2f}%")
            print(f"   📐 Trend direction: {trend_direction}")
            print(f"   🎯 Confidence interval: {confidence_interval[0]:.2f}% - {confidence_interval[1]:.2f}%")
            
            return self.trend_statistics
            
        except Exception as e:
            raise StatisticalAnalysisError(f"Failed to calculate trend statistics: {e}")

    def _calculate_trend_slope(self, data: List[CoverageMeasurement]) -> Tuple[float, float, str]:
        """Calculate trend slope using linear regression analysis."""
        try:
            if len(data) < 2:
                return 0.0, 0.0, "unknown"
            
            # Convert timestamps to numeric values (days since first measurement)
            first_timestamp = datetime.fromisoformat(data[0].timestamp.replace('Z', '+00:00'))
            x_values = []
            y_values = []
            
            for measurement in data:
                timestamp = datetime.fromisoformat(measurement.timestamp.replace('Z', '+00:00'))
                days_since_start = (timestamp - first_timestamp).total_seconds() / 86400  # Convert to days
                x_values.append(days_since_start)
                y_values.append(measurement.overall_coverage)
            
            # Use NumPy if available for more accurate calculations
            if HAS_NUMPY:
                slope, intercept = np.polyfit(x_values, y_values, 1)
                correlation_matrix = np.corrcoef(x_values, y_values)
                r_squared = correlation_matrix[0, 1] ** 2
            else:
                # Simple linear regression implementation
                n = len(x_values)
                sum_x = sum(x_values)
                sum_y = sum(y_values)
                sum_xy = sum(x * y for x, y in zip(x_values, y_values))
                sum_x_squared = sum(x * x for x in x_values)
                sum_y_squared = sum(y * y for y in y_values)
                
                # Calculate slope
                denominator = n * sum_x_squared - sum_x * sum_x
                if abs(denominator) < 1e-10:
                    slope = 0.0
                else:
                    slope = (n * sum_xy - sum_x * sum_y) / denominator
                
                # Calculate R-squared
                if n > 1:
                    ss_tot = sum_y_squared - (sum_y * sum_y) / n
                    y_mean = sum_y / n
                    ss_res = sum((y - (slope * x + (y_mean - slope * sum_x / n))) ** 2 
                                for x, y in zip(x_values, y_values))
                    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                else:
                    r_squared = 0.0
            
            # Determine trend direction
            if abs(slope) < 0.1:  # Threshold for "stable"
                trend_direction = "stable"
            elif slope > 0:
                trend_direction = "increasing"
            else:
                trend_direction = "decreasing"
            
            return slope, max(0.0, r_squared), trend_direction
            
        except Exception as e:
            print(f"⚠️  Warning: Failed to calculate trend slope: {e}")
            return 0.0, 0.0, "unknown"

    def detect_regression(self) -> RegressionAnalysis:
        """Detect coverage regression using statistical analysis and confidence intervals."""
        try:
            if len(self.historical_data) < 5:
                print("ℹ️  Insufficient data for regression detection (need at least 5 measurements)")
                return RegressionAnalysis(
                    regression_detected=False,
                    severity="none",
                    coverage_drop=0.0,
                    confidence_level=0.0,
                    recent_measurements=[],
                    baseline_average=0.0
                )
            
            # Get regression detection configuration
            regression_config = self.coverage_thresholds.get('global_settings', {}).get('regression_detection', {})
            min_drop_threshold = regression_config.get('min_drop_threshold', 2.0)
            confidence_level = regression_config.get('confidence_level', 0.95)
            
            # Analyze recent measurements vs. baseline
            recent_count = min(5, len(self.historical_data) // 3)  # Last 1/3 or 5 measurements
            baseline_count = len(self.historical_data) - recent_count
            
            if baseline_count < 2:
                baseline_count = len(self.historical_data) // 2
                recent_count = len(self.historical_data) - baseline_count
            
            baseline_measurements = self.historical_data[:baseline_count]
            recent_measurements = self.historical_data[-recent_count:]
            
            # Calculate baseline and recent averages
            baseline_coverage = [m.overall_coverage for m in baseline_measurements]
            recent_coverage = [m.overall_coverage for m in recent_measurements]
            
            baseline_average = statistics.mean(baseline_coverage)
            recent_average = statistics.mean(recent_coverage)
            
            coverage_drop = baseline_average - recent_average
            
            # Determine regression severity
            regression_detected = coverage_drop >= min_drop_threshold
            
            if coverage_drop >= 10.0:
                severity = "critical"
            elif coverage_drop >= 5.0:
                severity = "severe"
            elif coverage_drop >= 3.0:
                severity = "moderate"
            elif coverage_drop >= min_drop_threshold:
                severity = "minor"
            else:
                severity = "none"
            
            # Calculate statistical confidence of regression
            if len(baseline_coverage) > 1 and len(recent_coverage) > 1:
                baseline_std = statistics.stdev(baseline_coverage)
                recent_std = statistics.stdev(recent_coverage)
                
                # Simplified t-test approach for confidence
                pooled_std = math.sqrt((baseline_std**2 + recent_std**2) / 2)
                if pooled_std > 0:
                    t_statistic = abs(coverage_drop) / (pooled_std * math.sqrt(2))
                    # Approximate confidence (simplified)
                    statistical_confidence = min(0.99, 1.0 - (1.0 / (1.0 + t_statistic)))
                else:
                    statistical_confidence = 0.5
            else:
                statistical_confidence = 0.5
            
            # Generate recommendations
            recommendation = self._generate_regression_recommendation(severity, coverage_drop)
            
            # Identify affected areas (simplified analysis)
            affected_areas = []
            if self.current_measurement:
                if self.current_measurement.line_coverage < baseline_average:
                    affected_areas.append("line_coverage")
                if self.current_measurement.branch_coverage < 85.0:  # Branch threshold
                    affected_areas.append("branch_coverage")
                if self.current_measurement.critical_modules_coverage < 100.0:
                    affected_areas.append("critical_modules")
            
            self.regression_analysis = RegressionAnalysis(
                regression_detected=regression_detected,
                severity=severity,
                coverage_drop=coverage_drop,
                confidence_level=statistical_confidence,
                recent_measurements=recent_coverage,
                baseline_average=baseline_average,
                affected_areas=affected_areas,
                recommendation=recommendation
            )
            
            if regression_detected:
                print(f"🚨 Regression detected:")
                print(f"   📉 Coverage drop: {coverage_drop:.2f}%")
                print(f"   ⚠️  Severity: {severity}")
                print(f"   📊 Confidence: {statistical_confidence:.2f}")
            else:
                print(f"✅ No significant regression detected (drop: {coverage_drop:.2f}%)")
            
            return self.regression_analysis
            
        except Exception as e:
            raise RegressionDetectionError(f"Failed to detect regression: {e}")

    def _generate_regression_recommendation(self, severity: str, coverage_drop: float) -> str:
        """Generate actionable recommendations based on regression analysis."""
        if severity == "critical":
            return (f"Critical coverage regression detected ({coverage_drop:.1f}% drop). "
                   "Immediate action required: Review recent changes, add missing tests, "
                   "and consider blocking merge until coverage is restored.")
        elif severity == "severe":
            return (f"Severe coverage regression detected ({coverage_drop:.1f}% drop). "
                   "High priority: Review test coverage for recent changes and add "
                   "appropriate test cases before merge.")
        elif severity == "moderate":
            return (f"Moderate coverage regression detected ({coverage_drop:.1f}% drop). "
                   "Review: Ensure new code has adequate test coverage and consider "
                   "adding tests for uncovered areas.")
        elif severity == "minor":
            return (f"Minor coverage regression detected ({coverage_drop:.1f}% drop). "
                   "Monitor: Check if recent changes require additional test coverage.")
        else:
            return "Coverage levels are stable. Continue monitoring trends."

    def generate_quality_gate_alerts(self) -> List[QualityGateAlert]:
        """Generate quality gate alerts based on coverage analysis and thresholds."""
        try:
            self.quality_alerts = []
            
            if not self.current_measurement:
                print("ℹ️  No current measurement available for quality gate analysis")
                return self.quality_alerts
            
            # Check coverage threshold violations
            line_threshold = self.coverage_thresholds.get('global_settings', {}).get('overall_threshold', {}).get('line_coverage', 90.0)
            branch_threshold = self.coverage_thresholds.get('global_settings', {}).get('overall_threshold', {}).get('branch_coverage', 85.0)
            
            # Line coverage threshold alert
            if self.current_measurement.line_coverage < line_threshold:
                alert = QualityGateAlert(
                    alert_type="threshold_violation",
                    severity="error",
                    message=f"Line coverage {self.current_measurement.line_coverage:.2f}% below threshold {line_threshold}%",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    coverage_value=self.current_measurement.line_coverage,
                    threshold_value=line_threshold,
                    details={
                        "coverage_type": "line",
                        "build_number": self.current_measurement.build_number,
                        "commit_hash": self.current_measurement.commit_hash
                    }
                )
                self.quality_alerts.append(alert)
            
            # Branch coverage threshold alert
            if self.current_measurement.branch_coverage < branch_threshold:
                alert = QualityGateAlert(
                    alert_type="threshold_violation",
                    severity="warning",
                    message=f"Branch coverage {self.current_measurement.branch_coverage:.2f}% below threshold {branch_threshold}%",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    coverage_value=self.current_measurement.branch_coverage,
                    threshold_value=branch_threshold,
                    details={
                        "coverage_type": "branch",
                        "build_number": self.current_measurement.build_number,
                        "commit_hash": self.current_measurement.commit_hash
                    }
                )
                self.quality_alerts.append(alert)
            
            # Regression detection alert
            if self.regression_analysis and self.regression_analysis.regression_detected:
                severity_mapping = {
                    "critical": "critical",
                    "severe": "error",
                    "moderate": "warning",
                    "minor": "info"
                }
                
                alert = QualityGateAlert(
                    alert_type="regression_detected",
                    severity=severity_mapping.get(self.regression_analysis.severity, "warning"),
                    message=f"Coverage regression detected: {self.regression_analysis.coverage_drop:.2f}% drop ({self.regression_analysis.severity})",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    coverage_value=self.current_measurement.overall_coverage,
                    threshold_value=self.regression_analysis.baseline_average,
                    details={
                        "regression_severity": self.regression_analysis.severity,
                        "confidence_level": self.regression_analysis.confidence_level,
                        "affected_areas": self.regression_analysis.affected_areas,
                        "recommendation": self.regression_analysis.recommendation
                    }
                )
                self.quality_alerts.append(alert)
            
            # Coverage degradation trend alert
            if self.trend_statistics and self.trend_statistics.trend_direction == "decreasing":
                if abs(self.trend_statistics.trend_slope) > 0.5:  # Significant downward trend
                    alert = QualityGateAlert(
                        alert_type="coverage_degradation",
                        severity="warning",
                        message=f"Downward coverage trend detected (slope: {self.trend_statistics.trend_slope:.3f})",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        coverage_value=self.current_measurement.overall_coverage,
                        threshold_value=self.trend_statistics.mean,
                        details={
                            "trend_direction": self.trend_statistics.trend_direction,
                            "trend_slope": self.trend_statistics.trend_slope,
                            "r_squared": self.trend_statistics.r_squared,
                            "data_points": self.trend_statistics.data_points
                        }
                    )
                    self.quality_alerts.append(alert)
            
            print(f"🚦 Generated {len(self.quality_alerts)} quality gate alerts")
            for alert in self.quality_alerts:
                severity_emoji = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "critical": "🚨"}
                print(f"   {severity_emoji.get(alert.severity, '🔔')} {alert.alert_type}: {alert.message}")
            
            return self.quality_alerts
            
        except Exception as e:
            raise QualityGateAlertError(f"Failed to generate quality gate alerts: {e}")

    def save_historical_data(self) -> None:
        """Save historical coverage data with 30-day retention policy."""
        try:
            # Apply retention policy before saving
            self._apply_retention_policy()
            
            # Prepare data for serialization
            data = {
                "format_version": "1.0",
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "retention_days": 30,
                "measurements_count": len(self.historical_data),
                "measurements": [asdict(measurement) for measurement in self.historical_data]
            }
            
            # Write to historical data file
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Saved {len(self.historical_data)} measurements to {self.history_file}")
            
        except Exception as e:
            raise HistoricalDataError(f"Failed to save historical data: {e}")

    def generate_trend_report(self, output_format: str = "json") -> Path:
        """Generate comprehensive trend analysis report."""
        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            
            if output_format.lower() == "json":
                report_file = self.reports_dir / f"coverage_trend_analysis_{timestamp}.json"
                self._generate_json_trend_report(report_file)
            elif output_format.lower() == "html":
                report_file = self.reports_dir / f"coverage_trend_analysis_{timestamp}.html"
                self._generate_html_trend_report(report_file)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
            
            print(f"✅ Generated trend analysis report: {report_file}")
            return report_file
            
        except Exception as e:
            raise CoverageTrendException(f"Failed to generate trend report: {e}")

    def _generate_json_trend_report(self, output_file: Path) -> None:
        """Generate JSON format trend analysis report."""
        report_data = {
            "meta": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "analysis_period_days": self.trend_statistics.period_days if self.trend_statistics else 30,
                "total_measurements": len(self.historical_data),
                "coverage_thresholds": self.coverage_thresholds,
                "format_version": "1.0"
            },
            "current_measurement": asdict(self.current_measurement) if self.current_measurement else None,
            "trend_statistics": asdict(self.trend_statistics) if self.trend_statistics else None,
            "regression_analysis": asdict(self.regression_analysis) if self.regression_analysis else None,
            "quality_alerts": [asdict(alert) for alert in self.quality_alerts],
            "historical_summary": {
                "measurement_count": len(self.historical_data),
                "date_range": {
                    "start": self.historical_data[0].timestamp if self.historical_data else None,
                    "end": self.historical_data[-1].timestamp if self.historical_data else None
                },
                "coverage_range": {
                    "min": min(m.overall_coverage for m in self.historical_data) if self.historical_data else 0,
                    "max": max(m.overall_coverage for m in self.historical_data) if self.historical_data else 0
                }
            },
            "recommendations": self._generate_recommendations()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

    def _generate_html_trend_report(self, output_file: Path) -> None:
        """Generate HTML format trend analysis report with visualization."""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlyrigLoader Coverage Trend Analysis</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 6px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2563eb; }}
        .metric-label {{ font-size: 14px; color: #6b7280; margin-top: 5px; }}
        .alert {{ padding: 15px; margin: 10px 0; border-radius: 6px; }}
        .alert-critical {{ background: #fef2f2; border-left: 4px solid #dc2626; }}
        .alert-error {{ background: #fef2f2; border-left: 4px solid #dc2626; }}
        .alert-warning {{ background: #fffbeb; border-left: 4px solid #d97706; }}
        .alert-info {{ background: #eff6ff; border-left: 4px solid #2563eb; }}
        .section {{ margin: 30px 0; }}
        .section h2 {{ color: #1f2937; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e5e7eb; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        .trend-increasing {{ color: #059669; }}
        .trend-decreasing {{ color: #dc2626; }}
        .trend-stable {{ color: #6b7280; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 FlyrigLoader Coverage Trend Analysis</h1>
            <p>Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}</p>
        </div>
        
        <div class="section">
            <h2>📊 Current Coverage Metrics</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{self.current_measurement.overall_coverage:.1f}%</div>
                    <div class="metric-label">Overall Coverage</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{self.current_measurement.line_coverage:.1f}%</div>
                    <div class="metric-label">Line Coverage</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{self.current_measurement.branch_coverage:.1f}%</div>
                    <div class="metric-label">Branch Coverage</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{self.trend_statistics.trend_direction if self.trend_statistics else 'unknown'}</div>
                    <div class="metric-label">Trend Direction</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>🚦 Quality Gate Alerts</h2>
            {self._render_alerts_html()}
        </div>
        
        <div class="section">
            <h2>📈 Trend Analysis</h2>
            {self._render_trend_analysis_html()}
        </div>
        
        <div class="section">
            <h2>🔍 Regression Analysis</h2>
            {self._render_regression_analysis_html()}
        </div>
        
        <div class="section">
            <h2>💡 Recommendations</h2>
            {self._render_recommendations_html()}
        </div>
    </div>
</body>
</html>
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _render_alerts_html(self) -> str:
        """Render quality gate alerts as HTML."""
        if not self.quality_alerts:
            return "<p>✅ No quality gate alerts. All thresholds are met.</p>"
        
        html = ""
        for alert in self.quality_alerts:
            html += f"""
            <div class="alert alert-{alert.severity}">
                <strong>{alert.alert_type.replace('_', ' ').title()}:</strong> {alert.message}
                <br><small>Coverage: {alert.coverage_value:.2f}% | Threshold: {alert.threshold_value:.2f}%</small>
            </div>
            """
        return html

    def _render_trend_analysis_html(self) -> str:
        """Render trend analysis as HTML."""
        if not self.trend_statistics:
            return "<p>Insufficient data for trend analysis.</p>"
        
        trend_class = f"trend-{self.trend_statistics.trend_direction}"
        return f"""
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Mean Coverage</td><td>{self.trend_statistics.mean:.2f}%</td></tr>
            <tr><td>Median Coverage</td><td>{self.trend_statistics.median:.2f}%</td></tr>
            <tr><td>Standard Deviation</td><td>{self.trend_statistics.std_deviation:.2f}%</td></tr>
            <tr><td>Trend Direction</td><td><span class="{trend_class}">{self.trend_statistics.trend_direction}</span></td></tr>
            <tr><td>Trend Slope</td><td>{self.trend_statistics.trend_slope:.4f}</td></tr>
            <tr><td>R-Squared</td><td>{self.trend_statistics.r_squared:.3f}</td></tr>
            <tr><td>Confidence Interval (95%)</td><td>{self.trend_statistics.confidence_interval_95[0]:.2f}% - {self.trend_statistics.confidence_interval_95[1]:.2f}%</td></tr>
        </table>
        """

    def _render_regression_analysis_html(self) -> str:
        """Render regression analysis as HTML."""
        if not self.regression_analysis:
            return "<p>No regression analysis available.</p>"
        
        return f"""
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Regression Detected</td><td>{'Yes' if self.regression_analysis.regression_detected else 'No'}</td></tr>
            <tr><td>Severity</td><td>{self.regression_analysis.severity}</td></tr>
            <tr><td>Coverage Drop</td><td>{self.regression_analysis.coverage_drop:.2f}%</td></tr>
            <tr><td>Confidence Level</td><td>{self.regression_analysis.confidence_level:.2f}</td></tr>
            <tr><td>Baseline Average</td><td>{self.regression_analysis.baseline_average:.2f}%</td></tr>
            <tr><td>Affected Areas</td><td>{', '.join(self.regression_analysis.affected_areas) if self.regression_analysis.affected_areas else 'None'}</td></tr>
        </table>
        <p><strong>Recommendation:</strong> {self.regression_analysis.recommendation}</p>
        """

    def _render_recommendations_html(self) -> str:
        """Render recommendations as HTML."""
        recommendations = self._generate_recommendations()
        html = "<ul>"
        for recommendation in recommendations:
            html += f"<li>{recommendation}</li>"
        html += "</ul>"
        return html

    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis results."""
        recommendations = []
        
        if not self.current_measurement:
            recommendations.append("Run coverage analysis to generate current measurements")
            return recommendations
        
        # Coverage threshold recommendations
        line_threshold = self.coverage_thresholds.get('global_settings', {}).get('overall_threshold', {}).get('line_coverage', 90.0)
        if self.current_measurement.line_coverage < line_threshold:
            gap = line_threshold - self.current_measurement.line_coverage
            recommendations.append(f"Increase line coverage by {gap:.1f}% to meet {line_threshold}% threshold")
        
        # Regression recommendations
        if self.regression_analysis and self.regression_analysis.regression_detected:
            recommendations.append(self.regression_analysis.recommendation)
        
        # Trend recommendations
        if self.trend_statistics:
            if self.trend_statistics.trend_direction == "decreasing":
                recommendations.append("Address downward coverage trend by reviewing recent changes and adding tests")
            elif self.trend_statistics.std_deviation > 5.0:
                recommendations.append("High coverage variability detected - work on consistent test coverage practices")
        
        # Quality gate recommendations
        if self.quality_alerts:
            critical_alerts = [a for a in self.quality_alerts if a.severity == "critical"]
            if critical_alerts:
                recommendations.append("Address critical quality gate violations immediately before merge")
        
        # General recommendations
        if self.current_measurement.critical_modules_coverage < 100.0:
            recommendations.append("Ensure critical modules maintain 100% test coverage")
        
        if not recommendations:
            recommendations.append("Coverage levels are healthy - continue current testing practices")
        
        return recommendations

    def upload_artifacts_to_ci(self) -> bool:
        """Upload trend analysis artifacts to CI/CD system."""
        try:
            # Generate reports for CI/CD artifacts
            json_report = self.generate_trend_report("json")
            html_report = self.generate_trend_report("html")
            
            # GitHub Actions artifact preparation
            if os.environ.get('GITHUB_ACTIONS'):
                print("☁️  Preparing artifacts for GitHub Actions upload...")
                
                # Copy reports to artifacts directory
                import shutil
                shutil.copy2(json_report, self.artifacts_dir / "coverage_trend_analysis.json")
                shutil.copy2(html_report, self.artifacts_dir / "coverage_trend_analysis.html")
                
                # Create summary for GitHub Actions
                summary_file = self.artifacts_dir / "trend_analysis_summary.txt"
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(f"Coverage Trend Analysis Summary\n")
                    f.write(f"==============================\n\n")
                    f.write(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
                    if self.current_measurement:
                        f.write(f"Current Coverage: {self.current_measurement.overall_coverage:.2f}%\n")
                    if self.trend_statistics:
                        f.write(f"Trend Direction: {self.trend_statistics.trend_direction}\n")
                    if self.regression_analysis:
                        f.write(f"Regression Detected: {'Yes' if self.regression_analysis.regression_detected else 'No'}\n")
                    f.write(f"Quality Alerts: {len(self.quality_alerts)}\n")
                
                print(f"✅ Artifacts prepared in {self.artifacts_dir}")
                return True
            else:
                print("ℹ️  No CI/CD system detected for artifact upload")
                return False
                
        except Exception as e:
            print(f"⚠️  Warning: Failed to upload artifacts: {e}")
            return False

    def perform_comprehensive_analysis(self, coverage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform complete coverage trend analysis workflow."""
        try:
            print("🚀 Starting comprehensive coverage trend analysis...")
            
            # Load historical data
            self.load_historical_data()
            
            # Add current measurement
            self.add_current_measurement(coverage_data)
            
            # Calculate trend statistics
            self.calculate_trend_statistics()
            
            # Detect regression
            self.detect_regression()
            
            # Generate quality gate alerts
            self.generate_quality_gate_alerts()
            
            # Save updated historical data
            self.save_historical_data()
            
            # Generate reports
            json_report = self.generate_trend_report("json")
            html_report = self.generate_trend_report("html")
            
            # Prepare summary results
            results = {
                "analysis_complete": True,
                "current_coverage": self.current_measurement.overall_coverage if self.current_measurement else 0.0,
                "trend_direction": self.trend_statistics.trend_direction if self.trend_statistics else "unknown",
                "regression_detected": self.regression_analysis.regression_detected if self.regression_analysis else False,
                "quality_alerts_count": len(self.quality_alerts),
                "reports_generated": {
                    "json": str(json_report),
                    "html": str(html_report)
                },
                "recommendations": self._generate_recommendations(),
                "quality_gate_status": "passing" if not any(alert.severity in ["error", "critical"] for alert in self.quality_alerts) else "failing"
            }
            
            print("✅ Comprehensive coverage trend analysis completed successfully!")
            print(f"   📊 Current coverage: {results['current_coverage']:.2f}%")
            print(f"   📈 Trend: {results['trend_direction']}")
            print(f"   🚨 Alerts: {results['quality_alerts_count']}")
            print(f"   🚦 Quality gates: {results['quality_gate_status']}")
            
            return results
            
        except Exception as e:
            raise CoverageTrendException(f"Comprehensive analysis failed: {e}")


def main():
    """Main entry point for coverage trend analysis script."""
    parser = argparse.ArgumentParser(
        description="FlyrigLoader Coverage Trend Analysis Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze coverage from JSON report
  python analyze-coverage-trends.py --coverage-file coverage.json --full-analysis
  
  # Generate trend report only
  python analyze-coverage-trends.py --report-only --format html
  
  # Upload artifacts to CI/CD
  python analyze-coverage-trends.py --coverage-file coverage.json --upload-artifacts
  
  # Regression detection only
  python analyze-coverage-trends.py --coverage-file coverage.json --regression-only
        """
    )
    
    # Input options
    parser.add_argument(
        '--coverage-file',
        type=Path,
        help='Path to coverage JSON file from generate-coverage-reports.py'
    )
    parser.add_argument(
        '--project-root',
        type=Path,
        help='Project root directory (default: current working directory)'
    )
    
    # Analysis options
    parser.add_argument(
        '--full-analysis',
        action='store_true',
        help='Perform comprehensive trend analysis including regression detection'
    )
    parser.add_argument(
        '--regression-only',
        action='store_true',
        help='Perform only regression detection analysis'
    )
    parser.add_argument(
        '--trend-days',
        type=int,
        default=30,
        help='Number of days for trend analysis (default: 30)'
    )
    
    # Report options
    parser.add_argument(
        '--report-only',
        action='store_true',
        help='Generate reports from existing historical data'
    )
    parser.add_argument(
        '--format',
        choices=['json', 'html', 'both'],
        default='both',
        help='Report output format (default: both)'
    )
    
    # CI/CD integration options
    parser.add_argument(
        '--upload-artifacts',
        action='store_true',
        help='Upload analysis artifacts to CI/CD system'
    )
    parser.add_argument(
        '--quality-gate-check',
        action='store_true',
        help='Perform quality gate validation (exits with error on failure)'
    )
    
    # Other options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Configure verbosity
    if args.verbose:
        print("🔧 Running in verbose mode")
    
    try:
        # Initialize analyzer
        analyzer = CoverageTrendAnalyzer(project_root=args.project_root)
        
        # Load historical data
        analyzer.load_historical_data()
        
        if args.report_only:
            # Generate reports from existing data
            if args.format in ['json', 'both']:
                json_report = analyzer.generate_trend_report('json')
                print(f"📊 JSON report: {json_report}")
            
            if args.format in ['html', 'both']:
                html_report = analyzer.generate_trend_report('html')
                print(f"🌐 HTML report: {html_report}")
                
        elif args.coverage_file:
            # Load coverage data from file
            with open(args.coverage_file, 'r', encoding='utf-8') as f:
                coverage_data = json.load(f)
            
            if args.full_analysis:
                # Perform comprehensive analysis
                results = analyzer.perform_comprehensive_analysis(coverage_data)
                
                # Upload artifacts if requested
                if args.upload_artifacts:
                    analyzer.upload_artifacts_to_ci()
                
            elif args.regression_only:
                # Perform regression detection only
                analyzer.add_current_measurement(coverage_data)
                regression_result = analyzer.detect_regression()
                
                print(f"🔍 Regression analysis complete:")
                print(f"   Detected: {regression_result.regression_detected}")
                print(f"   Severity: {regression_result.severity}")
                print(f"   Coverage drop: {regression_result.coverage_drop:.2f}%")
                
                if regression_result.regression_detected and args.quality_gate_check:
                    print("❌ Regression detected - failing quality gate")
                    sys.exit(1)
            else:
                # Basic trend analysis
                analyzer.add_current_measurement(coverage_data)
                trend_stats = analyzer.calculate_trend_statistics(days=args.trend_days)
                
                print(f"📈 Trend analysis complete:")
                print(f"   Direction: {trend_stats.trend_direction}")
                print(f"   Mean coverage: {trend_stats.mean:.2f}%")
                print(f"   Trend slope: {trend_stats.trend_slope:.4f}")
        else:
            print("❌ Error: No coverage file specified and not in report-only mode")
            parser.print_help()
            sys.exit(1)
        
        # Perform quality gate check if requested
        if args.quality_gate_check and analyzer.quality_alerts:
            critical_alerts = [alert for alert in analyzer.quality_alerts 
                             if alert.severity in ['error', 'critical']]
            if critical_alerts:
                print(f"❌ Quality gate failure: {len(critical_alerts)} critical alerts")
                sys.exit(1)
        
        print("🎉 Coverage trend analysis completed successfully!")
        
    except (CoverageTrendException, HistoricalDataError, StatisticalAnalysisError, 
            RegressionDetectionError, QualityGateAlertError) as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()