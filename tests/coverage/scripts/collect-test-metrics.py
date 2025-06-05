#!/usr/bin/env python3
"""
Comprehensive Test Metrics Collection and Analytics System

Aggregates coverage data, performance benchmarks, test execution statistics, and quality 
indicators into unified reporting dashboard supporting automated quality assurance monitoring 
per Section 3.6.4 quality metrics dashboard integration and Section 0.2.5 infrastructure 
updates requirements.

This script serves as the central analytics engine for the flyrigloader test suite enhancement 
system, providing:

- Comprehensive test metrics aggregation from multiple data sources
- Automated metrics persistence with historical tracking and trend analysis
- Integration with pytest execution results for detailed test categorization
- Performance metrics collection correlating coverage with benchmark results
- Quality indicator calculation with execution time and regression detection
- Unified reporting with JSON export and visualization generation

Requirements Implementation:
- Section 3.6.4: Quality metrics dashboard integration with coverage trend tracking
- Section 0.2.5: Test execution time monitoring and failure rate analysis
- TST-PERF-001: Data loading SLA validation within 1s per 100MB correlation
- Section 2.1.12: Coverage Enhancement System with detailed reporting and visualization
- Section 4.1.1.5: Test execution workflow with automated quality gates

Author: FlyrigLoader Test Suite Enhancement Team
Created: 2024-12-19
License: MIT
"""

import argparse
import json
import logging
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

# Third-party imports for advanced analytics
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class TestExecutionMetrics:
    """
    Comprehensive test execution statistics with detailed categorization.
    
    Captures pytest execution results with comprehensive breakdown by test type,
    execution time analysis, and failure categorization per Section 4.1.1.5
    test execution workflow requirements.
    """
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    errors: int
    xfailed: int
    xpassed: int
    warnings: int
    execution_time: float
    test_categories: Dict[str, int]  # unit, integration, benchmark counts
    slowest_tests: List[Dict[str, Any]]
    failure_details: List[Dict[str, Any]]
    collection_time: float
    setup_time: float
    teardown_time: float


@dataclass 
class CoverageMetrics:
    """
    Comprehensive coverage analysis with module-level breakdown.
    
    Integrates with generate-coverage-reports.py results providing detailed
    coverage statistics with quality gate validation per TST-COV-001 and
    TST-COV-002 requirements.
    """
    overall_line_coverage: float
    overall_branch_coverage: float
    total_files: int
    total_statements: int
    covered_statements: int
    missing_statements: int
    total_branches: int
    covered_branches: int
    missing_branches: int
    module_coverage: Dict[str, Dict[str, Any]]
    critical_modules_coverage: Dict[str, float]
    quality_gate_status: str
    coverage_trend: List[Dict[str, Any]]
    threshold_violations: List[str]


@dataclass
class PerformanceMetrics:
    """
    Performance benchmark analysis with SLA validation.
    
    Integrates with check-performance-slas.py results providing comprehensive
    performance validation against TST-PERF-001 and TST-PERF-002 requirements.
    """
    data_loading_performance: Dict[str, float]  # size_mb -> time_seconds
    transformation_performance: Dict[str, float]  # rows_millions -> time_seconds
    sla_violations: List[Dict[str, Any]]
    performance_trends: Dict[str, List[float]]
    regression_alerts: List[str]
    benchmark_statistics: Dict[str, Dict[str, float]]
    overall_sla_status: str
    performance_score: float


@dataclass
class QualityIndicators:
    """
    Unified quality metrics with comprehensive analysis.
    
    Aggregates multiple quality dimensions providing holistic view of code
    health per Section 3.6.4 quality metrics dashboard integration.
    """
    overall_health_score: float  # 0-100 composite score
    coverage_score: float
    performance_score: float
    test_reliability_score: float
    trend_direction: str  # improving, stable, declining
    quality_gate_status: str  # PASS, WARN, FAIL
    recommendations: List[str]
    risk_indicators: List[str]
    improvement_opportunities: List[str]


@dataclass
class ComprehensiveTestReport:
    """
    Unified test metrics report with historical tracking.
    
    Provides complete view of test suite health with trend analysis and
    actionable insights per Section 2.1.12 Coverage Enhancement System.
    """
    timestamp: datetime
    execution_metrics: TestExecutionMetrics
    coverage_metrics: CoverageMetrics
    performance_metrics: PerformanceMetrics
    quality_indicators: QualityIndicators
    environment_info: Dict[str, Any]
    historical_comparison: Dict[str, Any]
    actionable_insights: List[str]
    dashboard_metrics: Dict[str, Any]


class TestMetricsCollector:
    """
    Comprehensive test metrics collection and analytics engine.
    
    Orchestrates data collection from multiple sources including coverage reports,
    performance benchmarks, and pytest execution results to provide unified
    quality dashboard integration per Section 3.6.4 requirements.
    """
    
    def __init__(self, 
                 project_root: Optional[str] = None,
                 coverage_reports_dir: Optional[str] = None,
                 performance_reports_dir: Optional[str] = None,
                 historical_data_dir: Optional[str] = None,
                 verbose: bool = False):
        """
        Initialize comprehensive test metrics collector.
        
        Args:
            project_root: Root directory of the flyrigloader project
            coverage_reports_dir: Directory containing coverage reports
            performance_reports_dir: Directory containing performance reports
            historical_data_dir: Directory for historical metrics storage
            verbose: Enable verbose logging output
        """
        self.verbose = verbose
        self.setup_logging()
        
        # Initialize directory paths
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent.parent.parent
        self.coverage_dir = self.project_root / "tests" / "coverage"
        self.scripts_dir = self.coverage_dir / "scripts"
        self.reports_dir = self.coverage_dir / "reports"
        self.metrics_dir = self.coverage_dir / "metrics"
        self.historical_dir = Path(historical_data_dir) if historical_data_dir else self.metrics_dir / "historical"
        
        # Coverage and performance specific directories
        self.coverage_reports_dir = Path(coverage_reports_dir) if coverage_reports_dir else self.project_root / "htmlcov"
        self.performance_reports_dir = Path(performance_reports_dir) if performance_reports_dir else self.coverage_dir / "benchmarks"
        
        # Configuration files
        self.thresholds_file = self.coverage_dir / "coverage-thresholds.json"
        self.quality_gates_file = self.coverage_dir / "quality-gates.yml"
        
        # Ensure required directories exist
        for directory in [self.reports_dir, self.metrics_dir, self.historical_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Initialize configuration
        self.thresholds_config = self._load_thresholds_config()
        
        # Runtime tracking
        self.collection_start_time = time.time()
        self.collection_stats = {
            'start_time': datetime.now(timezone.utc),
            'sources_processed': [],
            'warnings': [],
            'errors': []
        }
        
        self.logger.info(f"Initialized TestMetricsCollector")
        self.logger.info(f"Project root: {self.project_root}")
        self.logger.info(f"Coverage reports: {self.coverage_reports_dir}")
        self.logger.info(f"Performance reports: {self.performance_reports_dir}")

    def setup_logging(self) -> None:
        """Configure comprehensive logging for metrics collection."""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        log_file = self.project_root / "tests" / "coverage" / "test-metrics.log"
        
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
        self.logger = logging.getLogger("flyrigloader.metrics.collector")

    def _load_thresholds_config(self) -> Dict[str, Any]:
        """Load coverage thresholds configuration."""
        try:
            with open(self.thresholds_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.logger.info(f"Loaded thresholds configuration from {self.thresholds_file}")
            return config
        except FileNotFoundError:
            self.logger.warning(f"Thresholds configuration not found: {self.thresholds_file}")
            return self._get_default_thresholds()
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in thresholds configuration: {e}")
            return self._get_default_thresholds()

    def _get_default_thresholds(self) -> Dict[str, Any]:
        """Provide default threshold configuration."""
        return {
            "global_settings": {
                "overall_threshold": {
                    "line_coverage": 90.0,
                    "branch_coverage": 85.0
                }
            },
            "critical_modules": {
                "modules": {
                    "src/flyrigloader/api.py": {"line_coverage": 100.0, "branch_coverage": 100.0},
                    "src/flyrigloader/config/": {"line_coverage": 100.0, "branch_coverage": 100.0},
                    "src/flyrigloader/discovery/": {"line_coverage": 100.0, "branch_coverage": 100.0},
                    "src/flyrigloader/io/": {"line_coverage": 100.0, "branch_coverage": 100.0}
                }
            }
        }

    def collect_test_execution_metrics(self) -> TestExecutionMetrics:
        """
        Collect comprehensive pytest execution metrics.
        
        Analyzes pytest results including execution time, test categorization,
        and failure analysis per Section 4.1.1.5 test execution workflow.
        
        Returns:
            TestExecutionMetrics with comprehensive execution analysis
        """
        self.logger.info("Collecting test execution metrics")
        
        # Look for pytest result files
        junit_files = list(self.project_root.glob("**/pytest-results.xml"))
        if not junit_files:
            junit_files = list(self.project_root.glob("**/test-results.xml"))
        if not junit_files:
            junit_files = list(self.project_root.glob("**/.pytest_cache/"))
        
        if junit_files:
            return self._parse_junit_results(junit_files[0])
        else:
            # Execute pytest to collect fresh metrics
            return self._execute_pytest_metrics_collection()

    def _parse_junit_results(self, junit_file: Path) -> TestExecutionMetrics:
        """Parse JUnit XML results for test execution metrics."""
        try:
            tree = ET.parse(junit_file)
            root = tree.getroot()
            
            # Extract basic metrics
            total_tests = int(root.get('tests', 0))
            failures = int(root.get('failures', 0))
            errors = int(root.get('errors', 0))
            skipped = int(root.get('skipped', 0))
            execution_time = float(root.get('time', 0.0))
            
            passed_tests = total_tests - failures - errors - skipped
            
            # Analyze test cases for categorization
            test_categories = defaultdict(int)
            slowest_tests = []
            failure_details = []
            
            for testcase in root.findall('.//testcase'):
                test_name = testcase.get('name', '')
                test_time = float(testcase.get('time', 0.0))
                
                # Categorize tests
                if 'integration' in test_name.lower():
                    test_categories['integration'] += 1
                elif 'benchmark' in test_name.lower():
                    test_categories['benchmark'] += 1
                else:
                    test_categories['unit'] += 1
                
                # Track slow tests
                if test_time > 1.0:
                    slowest_tests.append({
                        'name': test_name,
                        'time': test_time,
                        'class': testcase.get('classname', '')
                    })
                
                # Track failures
                failure_elem = testcase.find('failure')
                error_elem = testcase.find('error')
                if failure_elem is not None or error_elem is not None:
                    failure_details.append({
                        'name': test_name,
                        'type': 'failure' if failure_elem is not None else 'error',
                        'message': (failure_elem or error_elem).get('message', ''),
                        'class': testcase.get('classname', '')
                    })
            
            # Sort slowest tests
            slowest_tests.sort(key=lambda x: x['time'], reverse=True)
            slowest_tests = slowest_tests[:10]  # Top 10 slowest
            
            return TestExecutionMetrics(
                total_tests=total_tests,
                passed_tests=passed_tests,
                failed_tests=failures,
                skipped_tests=skipped,
                errors=errors,
                xfailed=0,  # Not available in JUnit XML
                xpassed=0,  # Not available in JUnit XML
                warnings=0,  # Not available in JUnit XML
                execution_time=execution_time,
                test_categories=dict(test_categories),
                slowest_tests=slowest_tests,
                failure_details=failure_details,
                collection_time=0.0,  # Not available in JUnit XML
                setup_time=0.0,  # Not available in JUnit XML
                teardown_time=0.0  # Not available in JUnit XML
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse JUnit results: {e}")
            return self._get_default_test_metrics()

    def _execute_pytest_metrics_collection(self) -> TestExecutionMetrics:
        """Execute pytest to collect fresh test metrics."""
        self.logger.info("Executing pytest for fresh metrics collection")
        
        try:
            # Run pytest with comprehensive reporting
            cmd = [
                sys.executable, "-m", "pytest",
                "--tb=short",
                "--durations=10",
                "--junit-xml=pytest-results.xml",
                "-v",
                str(self.project_root / "tests")
            ]
            
            start_time = time.time()
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=self.project_root,
                timeout=300  # 5 minute timeout
            )
            execution_time = time.time() - start_time
            
            # Parse the generated JUnit file
            junit_file = self.project_root / "pytest-results.xml"
            if junit_file.exists():
                metrics = self._parse_junit_results(junit_file)
                metrics.execution_time = execution_time
                return metrics
            else:
                # Parse from stdout
                return self._parse_pytest_stdout(result.stdout, execution_time)
                
        except subprocess.TimeoutExpired:
            self.logger.error("Pytest execution timed out")
            return self._get_default_test_metrics()
        except Exception as e:
            self.logger.error(f"Failed to execute pytest: {e}")
            return self._get_default_test_metrics()

    def _parse_pytest_stdout(self, stdout: str, execution_time: float) -> TestExecutionMetrics:
        """Parse pytest stdout for test metrics."""
        lines = stdout.split('\n')
        
        # Initialize default values
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        skipped_tests = 0
        errors = 0
        
        # Look for summary line
        for line in lines:
            if "passed" in line and "failed" in line:
                # Parse summary line like "10 passed, 2 failed, 1 skipped in 5.23s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "passed" and i > 0:
                        passed_tests = int(parts[i-1])
                    elif part == "failed" and i > 0:
                        failed_tests = int(parts[i-1])
                    elif part == "skipped" and i > 0:
                        skipped_tests = int(parts[i-1])
                    elif part == "error" and i > 0:
                        errors = int(parts[i-1])
                break
        
        total_tests = passed_tests + failed_tests + skipped_tests + errors
        
        return TestExecutionMetrics(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            errors=errors,
            xfailed=0,
            xpassed=0,
            warnings=0,
            execution_time=execution_time,
            test_categories={'unit': total_tests},  # Default categorization
            slowest_tests=[],
            failure_details=[],
            collection_time=0.0,
            setup_time=0.0,
            teardown_time=0.0
        )

    def _get_default_test_metrics(self) -> TestExecutionMetrics:
        """Provide default test metrics when collection fails."""
        return TestExecutionMetrics(
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            skipped_tests=0,
            errors=0,
            xfailed=0,
            xpassed=0,
            warnings=0,
            execution_time=0.0,
            test_categories={'unit': 0, 'integration': 0, 'benchmark': 0},
            slowest_tests=[],
            failure_details=[],
            collection_time=0.0,
            setup_time=0.0,
            teardown_time=0.0
        )

    def collect_coverage_metrics(self) -> CoverageMetrics:
        """
        Collect comprehensive coverage metrics.
        
        Integrates with generate-coverage-reports.py results to provide detailed
        coverage analysis per Section 2.1.12 Coverage Enhancement System.
        
        Returns:
            CoverageMetrics with comprehensive coverage analysis
        """
        self.logger.info("Collecting coverage metrics")
        
        # Try to load existing coverage JSON report
        coverage_json_files = [
            self.coverage_reports_dir / "coverage.json",
            self.project_root / "coverage.json",
            self.reports_dir / "coverage.json"
        ]
        
        for json_file in coverage_json_files:
            if json_file.exists():
                return self._parse_coverage_json(json_file)
        
        # Try to parse coverage XML
        coverage_xml_files = [
            self.project_root / "coverage.xml",
            self.coverage_reports_dir / "coverage.xml"
        ]
        
        for xml_file in coverage_xml_files:
            if xml_file.exists():
                return self._parse_coverage_xml(xml_file)
        
        # Generate fresh coverage report
        return self._generate_coverage_metrics()

    def _parse_coverage_json(self, json_file: Path) -> CoverageMetrics:
        """Parse coverage JSON report for metrics."""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                coverage_data = json.load(f)
            
            # Extract summary metrics
            if 'coverage' in coverage_data and 'summary' in coverage_data['coverage']:
                summary = coverage_data['coverage']['summary']
                
                return CoverageMetrics(
                    overall_line_coverage=summary.get('overall_coverage_percentage', 0.0),
                    overall_branch_coverage=summary.get('branch_coverage_percentage', 0.0),
                    total_files=summary.get('total_files', 0),
                    total_statements=summary.get('total_statements', 0),
                    covered_statements=summary.get('covered_statements', 0),
                    missing_statements=summary.get('missing_statements', 0),
                    total_branches=summary.get('total_branches', 0),
                    covered_branches=summary.get('covered_branches', 0),
                    missing_branches=summary.get('missing_branches', 0),
                    module_coverage=coverage_data['coverage'].get('modules', {}),
                    critical_modules_coverage=self._extract_critical_module_coverage(coverage_data),
                    quality_gate_status=coverage_data['coverage'].get('quality_gates', {}).get('overall_threshold', {}).get('status', 'UNKNOWN'),
                    coverage_trend=self._get_coverage_trend(),
                    threshold_violations=self._identify_threshold_violations(coverage_data)
                )
            else:
                return self._get_default_coverage_metrics()
                
        except Exception as e:
            self.logger.error(f"Failed to parse coverage JSON: {e}")
            return self._get_default_coverage_metrics()

    def _parse_coverage_xml(self, xml_file: Path) -> CoverageMetrics:
        """Parse coverage XML report for metrics."""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Extract metrics from XML
            line_rate = float(root.get('line-rate', 0.0)) * 100
            branch_rate = float(root.get('branch-rate', 0.0)) * 100
            lines_covered = int(root.get('lines-covered', 0))
            lines_valid = int(root.get('lines-valid', 0))
            branches_covered = int(root.get('branches-covered', 0))
            branches_valid = int(root.get('branches-valid', 0))
            
            # Count files
            total_files = len(root.findall('.//class'))
            
            # Module-level analysis
            module_coverage = {}
            for package in root.findall('.//package'):
                package_name = package.get('name', '')
                for cls in package.findall('.//class'):
                    filename = cls.get('filename', '')
                    class_line_rate = float(cls.get('line-rate', 0.0)) * 100
                    class_branch_rate = float(cls.get('branch-rate', 0.0)) * 100
                    
                    module_coverage[filename] = {
                        'coverage_percentage': class_line_rate,
                        'branch_coverage_percentage': class_branch_rate
                    }
            
            return CoverageMetrics(
                overall_line_coverage=line_rate,
                overall_branch_coverage=branch_rate,
                total_files=total_files,
                total_statements=lines_valid,
                covered_statements=lines_covered,
                missing_statements=lines_valid - lines_covered,
                total_branches=branches_valid,
                covered_branches=branches_covered,
                missing_branches=branches_valid - branches_covered,
                module_coverage=module_coverage,
                critical_modules_coverage=self._extract_critical_module_coverage_from_xml(module_coverage),
                quality_gate_status=self._determine_quality_gate_status(line_rate, branch_rate),
                coverage_trend=self._get_coverage_trend(),
                threshold_violations=self._identify_xml_threshold_violations(line_rate, branch_rate)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse coverage XML: {e}")
            return self._get_default_coverage_metrics()

    def _generate_coverage_metrics(self) -> CoverageMetrics:
        """Generate fresh coverage metrics by running coverage analysis."""
        self.logger.info("Generating fresh coverage metrics")
        
        try:
            # Run coverage report generation
            generate_script = self.scripts_dir / "generate-coverage-reports.py"
            if generate_script.exists():
                result = subprocess.run([
                    sys.executable, str(generate_script),
                    "--output-dir", str(self.coverage_reports_dir)
                ], capture_output=True, text=True, cwd=self.project_root)
                
                if result.returncode == 0:
                    # Try to load the generated report
                    return self.collect_coverage_metrics()
            
            # Fallback to default metrics
            return self._get_default_coverage_metrics()
            
        except Exception as e:
            self.logger.error(f"Failed to generate coverage metrics: {e}")
            return self._get_default_coverage_metrics()

    def _extract_critical_module_coverage(self, coverage_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract coverage for critical modules."""
        critical_coverage = {}
        modules = coverage_data.get('coverage', {}).get('modules', {})
        
        critical_module_config = self.thresholds_config.get('critical_modules', {}).get('modules', {})
        
        for module_path in critical_module_config.keys():
            # Find matching module in coverage data
            for covered_module, module_data in modules.items():
                if module_path in covered_module:
                    critical_coverage[module_path] = module_data.get('coverage_percentage', 0.0)
                    break
        
        return critical_coverage

    def _extract_critical_module_coverage_from_xml(self, module_coverage: Dict[str, Any]) -> Dict[str, float]:
        """Extract critical module coverage from XML data."""
        critical_coverage = {}
        critical_module_config = self.thresholds_config.get('critical_modules', {}).get('modules', {})
        
        for module_path in critical_module_config.keys():
            for covered_module, module_data in module_coverage.items():
                if module_path in covered_module:
                    critical_coverage[module_path] = module_data.get('coverage_percentage', 0.0)
                    break
        
        return critical_coverage

    def _get_coverage_trend(self) -> List[Dict[str, Any]]:
        """Load historical coverage trend data."""
        trend_file = self.historical_dir / "coverage_trend.json"
        
        if trend_file.exists():
            try:
                with open(trend_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load coverage trend: {e}")
        
        return []

    def _identify_threshold_violations(self, coverage_data: Dict[str, Any]) -> List[str]:
        """Identify coverage threshold violations."""
        violations = []
        
        # Check overall threshold
        summary = coverage_data.get('coverage', {}).get('summary', {})
        overall_coverage = summary.get('overall_coverage_percentage', 0.0)
        overall_threshold = self.thresholds_config.get('global_settings', {}).get('overall_threshold', {}).get('line_coverage', 90.0)
        
        if overall_coverage < overall_threshold:
            violations.append(f"Overall coverage {overall_coverage:.1f}% below threshold {overall_threshold:.1f}%")
        
        # Check critical modules
        modules = coverage_data.get('coverage', {}).get('modules', {})
        critical_module_config = self.thresholds_config.get('critical_modules', {}).get('modules', {})
        
        for module_path, config in critical_module_config.items():
            required_coverage = config.get('line_coverage', 100.0)
            for covered_module, module_data in modules.items():
                if module_path in covered_module:
                    actual_coverage = module_data.get('coverage_percentage', 0.0)
                    if actual_coverage < required_coverage:
                        violations.append(f"Critical module {module_path}: {actual_coverage:.1f}% below required {required_coverage:.1f}%")
                    break
        
        return violations

    def _identify_xml_threshold_violations(self, line_rate: float, branch_rate: float) -> List[str]:
        """Identify threshold violations from XML data."""
        violations = []
        
        overall_threshold = self.thresholds_config.get('global_settings', {}).get('overall_threshold', {}).get('line_coverage', 90.0)
        branch_threshold = self.thresholds_config.get('global_settings', {}).get('overall_threshold', {}).get('branch_coverage', 85.0)
        
        if line_rate < overall_threshold:
            violations.append(f"Overall coverage {line_rate:.1f}% below threshold {overall_threshold:.1f}%")
        
        if branch_rate < branch_threshold:
            violations.append(f"Branch coverage {branch_rate:.1f}% below threshold {branch_threshold:.1f}%")
        
        return violations

    def _determine_quality_gate_status(self, line_rate: float, branch_rate: float) -> str:
        """Determine quality gate status based on coverage rates."""
        overall_threshold = self.thresholds_config.get('global_settings', {}).get('overall_threshold', {}).get('line_coverage', 90.0)
        branch_threshold = self.thresholds_config.get('global_settings', {}).get('overall_threshold', {}).get('branch_coverage', 85.0)
        
        if line_rate >= overall_threshold and branch_rate >= branch_threshold:
            return "PASS"
        else:
            return "FAIL"

    def _get_default_coverage_metrics(self) -> CoverageMetrics:
        """Provide default coverage metrics when collection fails."""
        return CoverageMetrics(
            overall_line_coverage=0.0,
            overall_branch_coverage=0.0,
            total_files=0,
            total_statements=0,
            covered_statements=0,
            missing_statements=0,
            total_branches=0,
            covered_branches=0,
            missing_branches=0,
            module_coverage={},
            critical_modules_coverage={},
            quality_gate_status="UNKNOWN",
            coverage_trend=[],
            threshold_violations=[]
        )

    def collect_performance_metrics(self) -> PerformanceMetrics:
        """
        Collect comprehensive performance metrics.
        
        Integrates with check-performance-slas.py results to provide detailed
        performance analysis per TST-PERF-001 and TST-PERF-002 requirements.
        
        Returns:
            PerformanceMetrics with comprehensive performance analysis
        """
        self.logger.info("Collecting performance metrics")
        
        # Look for existing performance reports
        performance_files = list(self.performance_reports_dir.glob("**/performance_sla_report_*.json"))
        
        if performance_files:
            # Use the most recent report
            latest_report = max(performance_files, key=lambda p: p.stat().st_mtime)
            return self._parse_performance_report(latest_report)
        else:
            # Generate fresh performance report
            return self._generate_performance_metrics()

    def _parse_performance_report(self, report_file: Path) -> PerformanceMetrics:
        """Parse performance SLA report for metrics."""
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                performance_data = json.load(f)
            
            # Extract performance metrics
            sla_results = performance_data.get('sla_results', [])
            
            data_loading_performance = {}
            transformation_performance = {}
            sla_violations = []
            
            for sla_result in sla_results:
                sla = sla_result.get('sla', {})
                operation_type = sla.get('operation_type', '')
                
                if operation_type == 'data_loading':
                    # Extract data loading performance
                    benchmark_results = sla_result.get('benchmark_results', [])
                    for result in benchmark_results:
                        data_size = result.get('data_size', 0)
                        execution_time = result.get('execution_time', 0)
                        if data_size > 0:
                            data_loading_performance[data_size] = execution_time
                
                elif operation_type == 'data_transformation':
                    # Extract transformation performance
                    benchmark_results = sla_result.get('benchmark_results', [])
                    for result in benchmark_results:
                        data_size = result.get('data_size', 0)  # millions of rows
                        execution_time = result.get('execution_time', 0)
                        if data_size > 0:
                            transformation_performance[data_size] = execution_time
                
                # Check for violations
                if not sla_result.get('passed', True):
                    sla_violations.append({
                        'operation_type': operation_type,
                        'violation_details': sla_result.get('violation_details', ''),
                        'performance_ratio': sla_result.get('performance_ratio', 0.0)
                    })
            
            # Calculate performance score
            performance_score = self._calculate_performance_score(sla_violations, len(sla_results))
            
            return PerformanceMetrics(
                data_loading_performance=data_loading_performance,
                transformation_performance=transformation_performance,
                sla_violations=sla_violations,
                performance_trends=self._get_performance_trends(),
                regression_alerts=self._identify_performance_regressions(performance_data),
                benchmark_statistics=self._extract_benchmark_statistics(sla_results),
                overall_sla_status=performance_data.get('overall_status', 'UNKNOWN'),
                performance_score=performance_score
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse performance report: {e}")
            return self._get_default_performance_metrics()

    def _generate_performance_metrics(self) -> PerformanceMetrics:
        """Generate fresh performance metrics by running SLA validation."""
        self.logger.info("Generating fresh performance metrics")
        
        try:
            # Run performance SLA check
            sla_script = self.scripts_dir / "check-performance-slas.py"
            if sla_script.exists():
                result = subprocess.run([
                    sys.executable, str(sla_script),
                    "--output-format", "json"
                ], capture_output=True, text=True, cwd=self.project_root)
                
                if result.returncode == 0:
                    # Try to collect the generated report
                    return self.collect_performance_metrics()
            
            # Fallback to default metrics
            return self._get_default_performance_metrics()
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance metrics: {e}")
            return self._get_default_performance_metrics()

    def _calculate_performance_score(self, violations: List[Dict[str, Any]], total_slas: int) -> float:
        """Calculate performance score based on SLA compliance."""
        if total_slas == 0:
            return 0.0
        
        violation_count = len(violations)
        compliance_rate = (total_slas - violation_count) / total_slas
        return compliance_rate * 100.0

    def _get_performance_trends(self) -> Dict[str, List[float]]:
        """Load historical performance trend data."""
        trend_file = self.historical_dir / "performance_trends.json"
        
        if trend_file.exists():
            try:
                with open(trend_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load performance trends: {e}")
        
        return {"data_loading": [], "transformation": []}

    def _identify_performance_regressions(self, performance_data: Dict[str, Any]) -> List[str]:
        """Identify performance regression alerts."""
        regressions = []
        
        sla_results = performance_data.get('sla_results', [])
        for sla_result in sla_results:
            if sla_result.get('regression_detected', False):
                operation_type = sla_result.get('sla', {}).get('operation_type', '')
                regressions.append(f"Performance regression detected in {operation_type}")
        
        return regressions

    def _extract_benchmark_statistics(self, sla_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Extract comprehensive benchmark statistics."""
        statistics = {}
        
        for sla_result in sla_results:
            operation_type = sla_result.get('sla', {}).get('operation_type', '')
            benchmark_results = sla_result.get('benchmark_results', [])
            
            if benchmark_results:
                execution_times = [result.get('execution_time', 0) for result in benchmark_results]
                
                statistics[operation_type] = {
                    'mean': statistics.mean(execution_times) if execution_times else 0.0,
                    'median': statistics.median(execution_times) if execution_times else 0.0,
                    'min': min(execution_times) if execution_times else 0.0,
                    'max': max(execution_times) if execution_times else 0.0,
                    'std_dev': statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0
                }
        
        return statistics

    def _get_default_performance_metrics(self) -> PerformanceMetrics:
        """Provide default performance metrics when collection fails."""
        return PerformanceMetrics(
            data_loading_performance={},
            transformation_performance={},
            sla_violations=[],
            performance_trends={"data_loading": [], "transformation": []},
            regression_alerts=[],
            benchmark_statistics={},
            overall_sla_status="UNKNOWN",
            performance_score=0.0
        )

    def calculate_quality_indicators(self, 
                                   execution_metrics: TestExecutionMetrics,
                                   coverage_metrics: CoverageMetrics,
                                   performance_metrics: PerformanceMetrics) -> QualityIndicators:
        """
        Calculate comprehensive quality indicators.
        
        Aggregates multiple quality dimensions into unified health metrics
        per Section 3.6.4 quality metrics dashboard integration.
        
        Args:
            execution_metrics: Test execution statistics
            coverage_metrics: Coverage analysis results
            performance_metrics: Performance benchmark results
            
        Returns:
            QualityIndicators with comprehensive quality analysis
        """
        self.logger.info("Calculating quality indicators")
        
        # Calculate individual quality scores
        coverage_score = self._calculate_coverage_score(coverage_metrics)
        performance_score = performance_metrics.performance_score
        test_reliability_score = self._calculate_test_reliability_score(execution_metrics)
        
        # Calculate overall health score (weighted average)
        overall_health_score = (
            coverage_score * 0.4 +  # 40% weight on coverage
            performance_score * 0.3 +  # 30% weight on performance
            test_reliability_score * 0.3  # 30% weight on test reliability
        )
        
        # Determine trend direction
        trend_direction = self._determine_trend_direction(coverage_metrics, performance_metrics)
        
        # Determine quality gate status
        quality_gate_status = self._determine_overall_quality_gate_status(
            coverage_metrics, performance_metrics, execution_metrics
        )
        
        # Generate recommendations
        recommendations = self._generate_quality_recommendations(
            coverage_metrics, performance_metrics, execution_metrics
        )
        
        # Identify risk indicators
        risk_indicators = self._identify_risk_indicators(
            coverage_metrics, performance_metrics, execution_metrics
        )
        
        # Identify improvement opportunities
        improvement_opportunities = self._identify_improvement_opportunities(
            coverage_metrics, performance_metrics, execution_metrics
        )
        
        return QualityIndicators(
            overall_health_score=overall_health_score,
            coverage_score=coverage_score,
            performance_score=performance_score,
            test_reliability_score=test_reliability_score,
            trend_direction=trend_direction,
            quality_gate_status=quality_gate_status,
            recommendations=recommendations,
            risk_indicators=risk_indicators,
            improvement_opportunities=improvement_opportunities
        )

    def _calculate_coverage_score(self, coverage_metrics: CoverageMetrics) -> float:
        """Calculate coverage quality score."""
        line_coverage = coverage_metrics.overall_line_coverage
        branch_coverage = coverage_metrics.overall_branch_coverage
        
        # Weight line and branch coverage
        coverage_score = (line_coverage * 0.7) + (branch_coverage * 0.3)
        
        # Apply penalties for critical module violations
        critical_penalty = 0
        for module, coverage in coverage_metrics.critical_modules_coverage.items():
            if coverage < 100.0:
                critical_penalty += (100.0 - coverage) * 0.1  # 0.1 point penalty per % missing
        
        return max(0.0, coverage_score - critical_penalty)

    def _calculate_test_reliability_score(self, execution_metrics: TestExecutionMetrics) -> float:
        """Calculate test reliability score."""
        if execution_metrics.total_tests == 0:
            return 0.0
        
        # Base reliability on pass rate
        pass_rate = execution_metrics.passed_tests / execution_metrics.total_tests
        reliability_score = pass_rate * 100.0
        
        # Apply penalties for errors and long execution times
        if execution_metrics.errors > 0:
            reliability_score -= execution_metrics.errors * 5.0  # 5 point penalty per error
        
        if execution_metrics.execution_time > 300:  # 5 minutes
            reliability_score -= (execution_metrics.execution_time - 300) / 60 * 2  # 2 points per minute over 5
        
        return max(0.0, min(100.0, reliability_score))

    def _determine_trend_direction(self, coverage_metrics: CoverageMetrics, performance_metrics: PerformanceMetrics) -> str:
        """Determine overall trend direction."""
        # Analyze coverage trend
        coverage_trend = coverage_metrics.coverage_trend
        if len(coverage_trend) >= 2:
            recent_coverage = coverage_trend[-1].get('coverage', 0)
            previous_coverage = coverage_trend[-2].get('coverage', 0)
            coverage_delta = recent_coverage - previous_coverage
        else:
            coverage_delta = 0
        
        # Analyze performance trends
        performance_trends = performance_metrics.performance_trends
        performance_delta = 0
        for operation, trend_data in performance_trends.items():
            if len(trend_data) >= 2:
                # Decreasing time is improvement (negative delta is good)
                performance_delta += (trend_data[-1] - trend_data[-2])
        
        # Combine trends
        if coverage_delta > 1.0 and performance_delta < 0.1:
            return "improving"
        elif coverage_delta < -1.0 or performance_delta > 0.1:
            return "declining"
        else:
            return "stable"

    def _determine_overall_quality_gate_status(self, 
                                             coverage_metrics: CoverageMetrics,
                                             performance_metrics: PerformanceMetrics,
                                             execution_metrics: TestExecutionMetrics) -> str:
        """Determine overall quality gate status."""
        # Check coverage gate
        coverage_pass = coverage_metrics.quality_gate_status == "PASS"
        
        # Check performance gate
        performance_pass = performance_metrics.overall_sla_status == "PASS"
        
        # Check test execution gate
        execution_pass = (execution_metrics.failed_tests == 0 and 
                         execution_metrics.errors == 0)
        
        if coverage_pass and performance_pass and execution_pass:
            return "PASS"
        elif not coverage_pass or not performance_pass or not execution_pass:
            return "FAIL"
        else:
            return "WARN"

    def _generate_quality_recommendations(self, 
                                        coverage_metrics: CoverageMetrics,
                                        performance_metrics: PerformanceMetrics,
                                        execution_metrics: TestExecutionMetrics) -> List[str]:
        """Generate actionable quality recommendations."""
        recommendations = []
        
        # Coverage recommendations
        if coverage_metrics.overall_line_coverage < 90.0:
            recommendations.append(
                f"Increase overall coverage from {coverage_metrics.overall_line_coverage:.1f}% to 90%+ by adding tests for uncovered code paths"
            )
        
        for module, coverage in coverage_metrics.critical_modules_coverage.items():
            if coverage < 100.0:
                recommendations.append(
                    f"Achieve 100% coverage for critical module {module} (currently {coverage:.1f}%)"
                )
        
        # Performance recommendations
        if len(performance_metrics.sla_violations) > 0:
            recommendations.append(
                "Address performance SLA violations to meet data loading and transformation requirements"
            )
        
        if len(performance_metrics.regression_alerts) > 0:
            recommendations.append(
                "Investigate performance regressions and optimize affected operations"
            )
        
        # Test execution recommendations
        if execution_metrics.execution_time > 300:  # 5 minutes
            recommendations.append(
                f"Optimize test execution time from {execution_metrics.execution_time:.1f}s to under 300s using parallel execution"
            )
        
        if execution_metrics.failed_tests > 0:
            recommendations.append(
                f"Fix {execution_metrics.failed_tests} failing tests to improve reliability"
            )
        
        return recommendations

    def _identify_risk_indicators(self, 
                                coverage_metrics: CoverageMetrics,
                                performance_metrics: PerformanceMetrics,
                                execution_metrics: TestExecutionMetrics) -> List[str]:
        """Identify potential risk indicators."""
        risks = []
        
        # Coverage risks
        if len(coverage_metrics.threshold_violations) > 0:
            risks.append("Coverage threshold violations may lead to untested code paths in production")
        
        # Performance risks
        if len(performance_metrics.sla_violations) > 0:
            risks.append("Performance SLA violations may impact user experience with large datasets")
        
        # Test execution risks
        if execution_metrics.errors > 0:
            risks.append("Test execution errors indicate potential infrastructure or configuration issues")
        
        if execution_metrics.total_tests == 0:
            risks.append("No tests executed - potential CI/CD pipeline configuration issue")
        
        return risks

    def _identify_improvement_opportunities(self, 
                                          coverage_metrics: CoverageMetrics,
                                          performance_metrics: PerformanceMetrics,
                                          execution_metrics: TestExecutionMetrics) -> List[str]:
        """Identify improvement opportunities."""
        opportunities = []
        
        # Coverage opportunities
        if coverage_metrics.overall_branch_coverage < coverage_metrics.overall_line_coverage:
            opportunities.append(
                "Improve branch coverage testing to match line coverage levels"
            )
        
        # Performance opportunities
        if PANDAS_AVAILABLE and len(performance_metrics.benchmark_statistics) > 0:
            for operation, stats in performance_metrics.benchmark_statistics.items():
                if stats.get('std_dev', 0) > stats.get('mean', 0) * 0.2:  # High variance
                    opportunities.append(
                        f"Reduce performance variance in {operation} operations for more predictable behavior"
                    )
        
        # Test execution opportunities
        if len(execution_metrics.test_categories.get('integration', 0)) < execution_metrics.test_categories.get('unit', 0) * 0.1:
            opportunities.append(
                "Increase integration test coverage to validate cross-module interactions"
            )
        
        return opportunities

    def persist_historical_data(self, report: ComprehensiveTestReport) -> None:
        """
        Persist metrics data for historical trend analysis.
        
        Stores comprehensive metrics with timestamp for long-term trend tracking
        per Section 0.2.5 infrastructure updates requirements.
        
        Args:
            report: Comprehensive test report to persist
        """
        self.logger.info("Persisting historical metrics data")
        
        try:
            # Persist coverage trend
            coverage_trend_file = self.historical_dir / "coverage_trend.json"
            coverage_trends = self._load_historical_data(coverage_trend_file, [])
            
            coverage_trends.append({
                'timestamp': report.timestamp.isoformat(),
                'coverage': report.coverage_metrics.overall_line_coverage,
                'branch_coverage': report.coverage_metrics.overall_branch_coverage
            })
            
            # Keep only last 100 entries
            coverage_trends = coverage_trends[-100:]
            self._save_historical_data(coverage_trend_file, coverage_trends)
            
            # Persist performance trends
            performance_trend_file = self.historical_dir / "performance_trends.json"
            performance_trends = self._load_historical_data(performance_trend_file, {"data_loading": [], "transformation": []})
            
            # Add current performance data
            current_loading_times = list(report.performance_metrics.data_loading_performance.values())
            current_transform_times = list(report.performance_metrics.transformation_performance.values())
            
            if current_loading_times:
                performance_trends["data_loading"].append(statistics.mean(current_loading_times))
                performance_trends["data_loading"] = performance_trends["data_loading"][-50:]  # Keep last 50
            
            if current_transform_times:
                performance_trends["transformation"].append(statistics.mean(current_transform_times))
                performance_trends["transformation"] = performance_trends["transformation"][-50:]  # Keep last 50
            
            self._save_historical_data(performance_trend_file, performance_trends)
            
            # Persist quality indicators trend
            quality_trend_file = self.historical_dir / "quality_trend.json"
            quality_trends = self._load_historical_data(quality_trend_file, [])
            
            quality_trends.append({
                'timestamp': report.timestamp.isoformat(),
                'health_score': report.quality_indicators.overall_health_score,
                'coverage_score': report.quality_indicators.coverage_score,
                'performance_score': report.quality_indicators.performance_score,
                'test_reliability_score': report.quality_indicators.test_reliability_score
            })
            
            quality_trends = quality_trends[-100:]  # Keep last 100
            self._save_historical_data(quality_trend_file, quality_trends)
            
            self.logger.info("Historical data persistence completed")
            
        except Exception as e:
            self.logger.error(f"Failed to persist historical data: {e}")

    def _load_historical_data(self, file_path: Path, default_value: Any) -> Any:
        """Load historical data from file with fallback to default."""
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load historical data from {file_path}: {e}")
        
        return default_value

    def _save_historical_data(self, file_path: Path, data: Any) -> None:
        """Save historical data to file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save historical data to {file_path}: {e}")

    def generate_comprehensive_report(self) -> ComprehensiveTestReport:
        """
        Generate comprehensive test metrics report.
        
        Orchestrates collection and analysis of all test metrics to provide
        unified dashboard integration per Section 3.6.4 requirements.
        
        Returns:
            ComprehensiveTestReport with unified metrics and analytics
        """
        self.logger.info("Generating comprehensive test metrics report")
        
        # Collect all metrics
        execution_metrics = self.collect_test_execution_metrics()
        self.collection_stats['sources_processed'].append('test_execution')
        
        coverage_metrics = self.collect_coverage_metrics()
        self.collection_stats['sources_processed'].append('coverage_analysis')
        
        performance_metrics = self.collect_performance_metrics()
        self.collection_stats['sources_processed'].append('performance_benchmarks')
        
        # Calculate quality indicators
        quality_indicators = self.calculate_quality_indicators(
            execution_metrics, coverage_metrics, performance_metrics
        )
        
        # Collect environment information
        environment_info = self._collect_environment_info()
        
        # Generate historical comparison
        historical_comparison = self._generate_historical_comparison(
            coverage_metrics, performance_metrics, quality_indicators
        )
        
        # Generate actionable insights
        actionable_insights = self._generate_actionable_insights(
            execution_metrics, coverage_metrics, performance_metrics, quality_indicators
        )
        
        # Generate dashboard metrics
        dashboard_metrics = self._generate_dashboard_metrics(
            execution_metrics, coverage_metrics, performance_metrics, quality_indicators
        )
        
        # Create comprehensive report
        report = ComprehensiveTestReport(
            timestamp=datetime.now(timezone.utc),
            execution_metrics=execution_metrics,
            coverage_metrics=coverage_metrics,
            performance_metrics=performance_metrics,
            quality_indicators=quality_indicators,
            environment_info=environment_info,
            historical_comparison=historical_comparison,
            actionable_insights=actionable_insights,
            dashboard_metrics=dashboard_metrics
        )
        
        # Persist historical data
        self.persist_historical_data(report)
        
        # Update collection statistics
        self.collection_stats['end_time'] = datetime.now(timezone.utc)
        self.collection_stats['duration'] = time.time() - self.collection_start_time
        
        self.logger.info(f"Comprehensive report generation completed in {self.collection_stats['duration']:.2f}s")
        
        return report

    def _collect_environment_info(self) -> Dict[str, Any]:
        """Collect comprehensive environment information."""
        import platform
        
        environment_info = {
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'collection_timestamp': datetime.now(timezone.utc).isoformat(),
            'project_root': str(self.project_root),
            'collection_duration': time.time() - self.collection_start_time
        }
        
        # Add CI/CD environment variables if available
        ci_vars = ['CI', 'GITHUB_ACTIONS', 'BUILD_NUMBER', 'BRANCH_NAME', 'COMMIT_SHA']
        for var in ci_vars:
            environment_info[var.lower()] = os.environ.get(var, 'not_set')
        
        return environment_info

    def _generate_historical_comparison(self, 
                                      coverage_metrics: CoverageMetrics,
                                      performance_metrics: PerformanceMetrics,
                                      quality_indicators: QualityIndicators) -> Dict[str, Any]:
        """Generate historical comparison analysis."""
        comparison = {
            'coverage_trend': 'no_data',
            'performance_trend': 'no_data',
            'quality_trend': 'no_data',
            'significant_changes': []
        }
        
        # Analyze coverage trend
        coverage_trend = coverage_metrics.coverage_trend
        if len(coverage_trend) >= 2:
            recent = coverage_trend[-1].get('coverage', 0)
            previous = coverage_trend[-2].get('coverage', 0)
            delta = recent - previous
            
            if abs(delta) > 1.0:  # Significant change
                comparison['coverage_trend'] = 'improving' if delta > 0 else 'declining'
                comparison['significant_changes'].append(
                    f"Coverage changed by {delta:+.1f}% from previous measurement"
                )
            else:
                comparison['coverage_trend'] = 'stable'
        
        # Analyze performance trend
        performance_trends = performance_metrics.performance_trends
        if any(len(trend) >= 2 for trend in performance_trends.values()):
            comparison['performance_trend'] = quality_indicators.trend_direction
        
        # Analyze overall quality trend
        quality_trend_file = self.historical_dir / "quality_trend.json"
        quality_trends = self._load_historical_data(quality_trend_file, [])
        
        if len(quality_trends) >= 2:
            recent_health = quality_trends[-1].get('health_score', 0)
            previous_health = quality_trends[-2].get('health_score', 0)
            delta = recent_health - previous_health
            
            if abs(delta) > 5.0:  # Significant change
                comparison['quality_trend'] = 'improving' if delta > 0 else 'declining'
                comparison['significant_changes'].append(
                    f"Overall health score changed by {delta:+.1f} points"
                )
            else:
                comparison['quality_trend'] = 'stable'
        
        return comparison

    def _generate_actionable_insights(self, 
                                    execution_metrics: TestExecutionMetrics,
                                    coverage_metrics: CoverageMetrics,
                                    performance_metrics: PerformanceMetrics,
                                    quality_indicators: QualityIndicators) -> List[str]:
        """Generate prioritized actionable insights."""
        insights = []
        
        # Priority 1: Critical failures
        if execution_metrics.failed_tests > 0 or execution_metrics.errors > 0:
            insights.append(
                f"CRITICAL: Fix {execution_metrics.failed_tests + execution_metrics.errors} failing/error tests immediately"
            )
        
        # Priority 2: Coverage gaps
        if len(coverage_metrics.threshold_violations) > 0:
            insights.append(
                f"HIGH: Address {len(coverage_metrics.threshold_violations)} coverage threshold violations"
            )
        
        # Priority 3: Performance issues
        if len(performance_metrics.sla_violations) > 0:
            insights.append(
                f"HIGH: Resolve {len(performance_metrics.sla_violations)} performance SLA violations"
            )
        
        # Priority 4: Quality improvements
        if quality_indicators.overall_health_score < 80.0:
            insights.append(
                f"MEDIUM: Overall health score {quality_indicators.overall_health_score:.1f}% needs improvement"
            )
        
        # Priority 5: Optimization opportunities
        if execution_metrics.execution_time > 180:  # 3 minutes
            insights.append(
                f"LOW: Consider optimizing test execution time ({execution_metrics.execution_time:.1f}s)"
            )
        
        return insights

    def _generate_dashboard_metrics(self, 
                                  execution_metrics: TestExecutionMetrics,
                                  coverage_metrics: CoverageMetrics,
                                  performance_metrics: PerformanceMetrics,
                                  quality_indicators: QualityIndicators) -> Dict[str, Any]:
        """Generate metrics optimized for dashboard visualization."""
        return {
            'summary_cards': {
                'overall_health': {
                    'value': quality_indicators.overall_health_score,
                    'unit': '%',
                    'status': quality_indicators.quality_gate_status,
                    'trend': quality_indicators.trend_direction
                },
                'test_pass_rate': {
                    'value': (execution_metrics.passed_tests / execution_metrics.total_tests * 100) if execution_metrics.total_tests > 0 else 0,
                    'unit': '%',
                    'status': 'PASS' if execution_metrics.failed_tests == 0 else 'FAIL',
                    'trend': 'stable'
                },
                'coverage_rate': {
                    'value': coverage_metrics.overall_line_coverage,
                    'unit': '%',
                    'status': coverage_metrics.quality_gate_status,
                    'trend': quality_indicators.trend_direction
                },
                'performance_score': {
                    'value': performance_metrics.performance_score,
                    'unit': '%',
                    'status': performance_metrics.overall_sla_status,
                    'trend': quality_indicators.trend_direction
                }
            },
            'detailed_metrics': {
                'test_execution': {
                    'total_tests': execution_metrics.total_tests,
                    'passed': execution_metrics.passed_tests,
                    'failed': execution_metrics.failed_tests,
                    'execution_time': execution_metrics.execution_time,
                    'categories': execution_metrics.test_categories
                },
                'coverage_breakdown': {
                    'line_coverage': coverage_metrics.overall_line_coverage,
                    'branch_coverage': coverage_metrics.overall_branch_coverage,
                    'total_statements': coverage_metrics.total_statements,
                    'violations': len(coverage_metrics.threshold_violations)
                },
                'performance_breakdown': {
                    'sla_violations': len(performance_metrics.sla_violations),
                    'regression_alerts': len(performance_metrics.regression_alerts),
                    'benchmark_count': len(performance_metrics.benchmark_statistics)
                }
            },
            'alerts': {
                'critical': quality_indicators.risk_indicators,
                'warnings': coverage_metrics.threshold_violations + performance_metrics.regression_alerts,
                'recommendations': quality_indicators.recommendations[:5]  # Top 5
            }
        }

    def export_report(self, report: ComprehensiveTestReport, output_format: str = "json", output_path: Optional[str] = None) -> str:
        """
        Export comprehensive test report in specified format.
        
        Supports JSON, HTML, and CSV export formats for diverse integration
        requirements per Section 2.1.12 detailed reporting and visualization.
        
        Args:
            report: Comprehensive test report to export
            output_format: Export format (json, html, csv)
            output_path: Custom output path (optional)
            
        Returns:
            Path to exported report file
        """
        self.logger.info(f"Exporting comprehensive report in {output_format} format")
        
        timestamp_str = report.timestamp.strftime("%Y%m%d_%H%M%S")
        
        if output_format.lower() == "json":
            return self._export_json_report(report, timestamp_str, output_path)
        elif output_format.lower() == "html":
            return self._export_html_report(report, timestamp_str, output_path)
        elif output_format.lower() == "csv":
            return self._export_csv_report(report, timestamp_str, output_path)
        else:
            raise ValueError(f"Unsupported export format: {output_format}")

    def _export_json_report(self, report: ComprehensiveTestReport, timestamp_str: str, output_path: Optional[str]) -> str:
        """Export report as comprehensive JSON."""
        if output_path:
            report_path = Path(output_path)
        else:
            report_path = self.reports_dir / f"comprehensive_test_metrics_{timestamp_str}.json"
        
        try:
            # Convert dataclasses to dictionaries
            report_dict = asdict(report)
            
            # Handle datetime serialization
            report_dict['timestamp'] = report.timestamp.isoformat()
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, indent=2, default=str)
            
            self.logger.info(f"JSON report exported to {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Failed to export JSON report: {e}")
            raise

    def _export_html_report(self, report: ComprehensiveTestReport, timestamp_str: str, output_path: Optional[str]) -> str:
        """Export report as comprehensive HTML dashboard."""
        if output_path:
            report_path = Path(output_path)
        else:
            report_path = self.reports_dir / f"comprehensive_test_metrics_{timestamp_str}.html"
        
        try:
            html_content = self._generate_html_dashboard(report)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"HTML report exported to {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Failed to export HTML report: {e}")
            raise

    def _export_csv_report(self, report: ComprehensiveTestReport, timestamp_str: str, output_path: Optional[str]) -> str:
        """Export report as CSV for data analysis."""
        if not PANDAS_AVAILABLE:
            raise RuntimeError("Pandas is required for CSV export")
        
        if output_path:
            report_path = Path(output_path)
        else:
            report_path = self.reports_dir / f"comprehensive_test_metrics_{timestamp_str}.csv"
        
        try:
            # Create summary data for CSV
            csv_data = {
                'timestamp': [report.timestamp.isoformat()],
                'overall_health_score': [report.quality_indicators.overall_health_score],
                'coverage_score': [report.quality_indicators.coverage_score],
                'performance_score': [report.quality_indicators.performance_score],
                'test_reliability_score': [report.quality_indicators.test_reliability_score],
                'total_tests': [report.execution_metrics.total_tests],
                'passed_tests': [report.execution_metrics.passed_tests],
                'failed_tests': [report.execution_metrics.failed_tests],
                'overall_line_coverage': [report.coverage_metrics.overall_line_coverage],
                'overall_branch_coverage': [report.coverage_metrics.overall_branch_coverage],
                'sla_violations': [len(report.performance_metrics.sla_violations)],
                'quality_gate_status': [report.quality_indicators.quality_gate_status],
                'trend_direction': [report.quality_indicators.trend_direction]
            }
            
            df = pd.DataFrame(csv_data)
            df.to_csv(report_path, index=False)
            
            self.logger.info(f"CSV report exported to {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Failed to export CSV report: {e}")
            raise

    def _generate_html_dashboard(self, report: ComprehensiveTestReport) -> str:
        """Generate comprehensive HTML dashboard."""
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlyrigLoader Test Metrics Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .status-pass {{ color: #4CAF50; }}
        .status-fail {{ color: #f44336; }}
        .status-warn {{ color: #ff9800; }}
        .section {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .insights-list {{
            list-style: none;
            padding: 0;
        }}
        .insights-list li {{
            padding: 10px;
            margin: 5px 0;
            border-left: 4px solid #667eea;
            background: #f8f9fa;
            border-radius: 4px;
        }}
        .recommendations {{
            background: #e3f2fd;
            border-left: 4px solid #2196F3;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
        }}
        .risk-indicator {{
            background: #ffebee;
            border-left: 4px solid #f44336;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>FlyrigLoader Test Metrics Dashboard</h1>
        <p>Generated on {report.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        <p>Overall Health Score: <span class="metric-value status-{report.quality_indicators.quality_gate_status.lower()}">{report.quality_indicators.overall_health_score:.1f}%</span></p>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <h3>Test Execution</h3>
            <div class="metric-value status-{'pass' if report.execution_metrics.failed_tests == 0 else 'fail'}">
                {report.execution_metrics.passed_tests}/{report.execution_metrics.total_tests}
            </div>
            <p>Tests Passed</p>
        </div>
        
        <div class="metric-card">
            <h3>Code Coverage</h3>
            <div class="metric-value status-{report.coverage_metrics.quality_gate_status.lower()}">
                {report.coverage_metrics.overall_line_coverage:.1f}%
            </div>
            <p>Line Coverage</p>
        </div>
        
        <div class="metric-card">
            <h3>Performance Score</h3>
            <div class="metric-value status-{report.performance_metrics.overall_sla_status.lower()}">
                {report.performance_metrics.performance_score:.1f}%
            </div>
            <p>SLA Compliance</p>
        </div>
        
        <div class="metric-card">
            <h3>Quality Gate</h3>
            <div class="metric-value status-{report.quality_indicators.quality_gate_status.lower()}">
                {report.quality_indicators.quality_gate_status}
            </div>
            <p>Overall Status</p>
        </div>
    </div>
    
    <div class="section">
        <h2>Actionable Insights</h2>
        <ul class="insights-list">
"""
        
        for insight in report.actionable_insights:
            html_template += f"<li>{insight}</li>"
        
        html_template += """
        </ul>
    </div>
    
    <div class="section">
        <h2>Quality Recommendations</h2>
"""
        
        for recommendation in report.quality_indicators.recommendations:
            html_template += f'<div class="recommendations">{recommendation}</div>'
        
        html_template += """
    </div>
    
    <div class="section">
        <h2>Risk Indicators</h2>
"""
        
        for risk in report.quality_indicators.risk_indicators:
            html_template += f'<div class="risk-indicator">{risk}</div>'
        
        html_template += f"""
    </div>
    
    <div class="section">
        <h2>Detailed Metrics</h2>
        <h3>Test Execution Details</h3>
        <p><strong>Total Tests:</strong> {report.execution_metrics.total_tests}</p>
        <p><strong>Passed:</strong> {report.execution_metrics.passed_tests}</p>
        <p><strong>Failed:</strong> {report.execution_metrics.failed_tests}</p>
        <p><strong>Skipped:</strong> {report.execution_metrics.skipped_tests}</p>
        <p><strong>Execution Time:</strong> {report.execution_metrics.execution_time:.1f}s</p>
        
        <h3>Coverage Details</h3>
        <p><strong>Line Coverage:</strong> {report.coverage_metrics.overall_line_coverage:.1f}%</p>
        <p><strong>Branch Coverage:</strong> {report.coverage_metrics.overall_branch_coverage:.1f}%</p>
        <p><strong>Total Statements:</strong> {report.coverage_metrics.total_statements:,}</p>
        <p><strong>Covered Statements:</strong> {report.coverage_metrics.covered_statements:,}</p>
        
        <h3>Performance Details</h3>
        <p><strong>SLA Violations:</strong> {len(report.performance_metrics.sla_violations)}</p>
        <p><strong>Regression Alerts:</strong> {len(report.performance_metrics.regression_alerts)}</p>
        <p><strong>Overall SLA Status:</strong> {report.performance_metrics.overall_sla_status}</p>
    </div>
    
    <div class="section">
        <h2>Environment Information</h2>
        <p><strong>Python Version:</strong> {report.environment_info.get('python_version', 'Unknown')}</p>
        <p><strong>Platform:</strong> {report.environment_info.get('platform', 'Unknown')}</p>
        <p><strong>Collection Duration:</strong> {report.environment_info.get('collection_duration', 0):.1f}s</p>
    </div>
</body>
</html>
"""
        
        return html_template


def main() -> int:
    """
    Main entry point for comprehensive test metrics collection.
    
    Implements command-line interface for complete test metrics aggregation
    with quality dashboard integration per Section 3.6.4 requirements.
    
    Returns:
        Exit code: 0 for success, 1 for failure
    """
    parser = argparse.ArgumentParser(
        description="Comprehensive Test Metrics Collection and Analytics System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                     # Collect all metrics with default settings
  %(prog)s --verbose                           # Enable verbose logging
  %(prog)s --output-format html                # Generate HTML dashboard
  %(prog)s --export-path ./reports/           # Custom export directory
  %(prog)s --historical-analysis              # Include historical trend analysis

Output Formats:
  - JSON: Comprehensive programmatic data for API integration
  - HTML: Interactive dashboard for development team visualization
  - CSV: Data analysis export for statistical processing

Quality Metrics:
  The script aggregates metrics from multiple sources:
  - Test execution statistics (pytest results)
  - Coverage analysis (pytest-cov integration)
  - Performance benchmarks (pytest-benchmark results)  
  - Quality indicators (composite health scoring)

Integration:
  Designed for CI/CD pipeline integration with automated quality gates
  and dashboard integration per Section 3.6.4 requirements.
        """
    )
    
    parser.add_argument(
        '--project-root',
        help='Root directory of the flyrigloader project',
        default=None
    )
    
    parser.add_argument(
        '--coverage-reports-dir',
        help='Directory containing coverage reports (default: htmlcov)',
        default=None
    )
    
    parser.add_argument(
        '--performance-reports-dir', 
        help='Directory containing performance reports (default: tests/coverage/benchmarks)',
        default=None
    )
    
    parser.add_argument(
        '--historical-data-dir',
        help='Directory for historical metrics storage (default: tests/coverage/metrics/historical)',
        default=None
    )
    
    parser.add_argument(
        '--output-format',
        choices=['json', 'html', 'csv'],
        default='json',
        help='Export format for comprehensive report'
    )
    
    parser.add_argument(
        '--export-path',
        help='Custom path for exported report',
        default=None
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging output'
    )
    
    parser.add_argument(
        '--historical-analysis',
        action='store_true',
        help='Include comprehensive historical trend analysis'
    )
    
    parser.add_argument(
        '--fail-on-quality-gate',
        action='store_true',
        help='Exit with non-zero code if quality gates fail'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='FlyrigLoader Test Metrics Collector 1.0.0'
    )

    args = parser.parse_args()

    # Initialize and run the comprehensive metrics collector
    try:
        collector = TestMetricsCollector(
            project_root=args.project_root,
            coverage_reports_dir=args.coverage_reports_dir,
            performance_reports_dir=args.performance_reports_dir,
            historical_data_dir=args.historical_data_dir,
            verbose=args.verbose
        )
        
        # Generate comprehensive report
        start_time = time.time()
        report = collector.generate_comprehensive_report()
        collection_time = time.time() - start_time
        
        # Export report
        report_path = collector.export_report(report, args.output_format, args.export_path)
        
        # Summary output
        collector.logger.info(f"Test metrics collection completed in {collection_time:.2f}s")
        collector.logger.info(f"Overall health score: {report.quality_indicators.overall_health_score:.1f}%")
        collector.logger.info(f"Quality gate status: {report.quality_indicators.quality_gate_status}")
        collector.logger.info(f"Report exported to: {report_path}")
        
        # Quality gate enforcement
        if args.fail_on_quality_gate and report.quality_indicators.quality_gate_status == "FAIL":
            collector.logger.error("Exiting with failure code due to quality gate failure")
            return 1
        
        # Success summary
        collector.logger.info("Test metrics collection completed successfully")
        return 0
        
    except KeyboardInterrupt:
        print("\nTest metrics collection interrupted by user")
        return 1
    except Exception as e:
        print(f"Fatal error in test metrics collection: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())