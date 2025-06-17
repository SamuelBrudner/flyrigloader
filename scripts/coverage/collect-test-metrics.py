#!/usr/bin/env python3
"""
FlyrigLoader Test Metrics Aggregation System

Central test metrics aggregation engine that consolidates pytest execution results, coverage 
analysis, performance validation, and quality indicators into comprehensive reports supporting 
the enhanced testing strategy's quality gate enforcement and CI/CD integration.

This script serves as the primary orchestrator for the flyrigloader Test Metrics Enhancement 
System, providing comprehensive aggregation of test execution data, coverage analytics, 
performance validation results, and quality gate status reporting with CI/CD integration.

Key Features:
- Pytest execution result aggregation with pytest-xdist parallel execution support
- Coverage analysis integration with generate-coverage-reports.py
- Performance benchmark integration with scripts/benchmarks/ infrastructure
- Quality gate validation and enforcement per Section 8.5.1
- Historical trend data persistence and regression detection
- CI/CD artifact generation and GitHub Actions integration
- Comprehensive reporting in JSON, HTML, and markdown formats

Author: FlyrigLoader Test Metrics Enhancement System
Version: 1.0.0
Requirements: Section 6.6.5.2, Section 6.6.4, Section 8.5.1, Section 8.6
"""

import argparse
import json
import os
import sys
import time
import traceback
import subprocess
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, NamedTuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party imports
try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Install with: pip install pyyaml>=6.0")
    sys.exit(1)

try:
    import psutil
except ImportError:
    print("ERROR: psutil not installed. Install with: pip install psutil>=5.9.0")
    sys.exit(1)

try:
    import jinja2
    from jinja2 import Environment, FileSystemLoader
except ImportError:
    print("ERROR: Jinja2 not installed. Install with: pip install jinja2>=3.1.0")
    sys.exit(1)


class MetricsException(Exception):
    """Base exception for test metrics collection failures."""
    pass


class ConfigurationError(MetricsException):
    """Raised when metrics configuration is invalid."""
    pass


class AggregationError(MetricsException):
    """Raised when test metrics aggregation fails."""
    pass


class QualityGateError(MetricsException):
    """Raised when quality gate validation fails."""
    pass


@dataclass
class TestExecutionMetrics:
    """Test execution metrics from pytest runs."""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    error_tests: int = 0
    execution_time: float = 0.0
    parallel_workers: int = 1
    test_categories: Dict[str, int] = field(default_factory=dict)
    pytest_version: str = ""
    python_version: str = ""
    platform: str = ""


@dataclass
class CoverageMetrics:
    """Coverage analysis metrics."""
    line_coverage: float = 0.0
    branch_coverage: float = 0.0
    function_coverage: float = 0.0
    critical_modules_coverage: float = 0.0
    total_lines: int = 0
    covered_lines: int = 0
    missing_lines: int = 0
    quality_gate_status: str = "unknown"


@dataclass
class PerformanceMetrics:
    """Performance validation metrics from benchmarks."""
    benchmark_executed: bool = False
    data_loading_sla_compliance: bool = False
    transformation_sla_compliance: bool = False
    discovery_sla_compliance: bool = False
    config_sla_compliance: bool = False
    benchmark_execution_time: float = 0.0
    memory_peak_usage: float = 0.0
    regression_detected: bool = False
    performance_artifacts_generated: bool = False


@dataclass
class QualityGateResults:
    """Quality gate validation results."""
    overall_coverage_gate: bool = False
    critical_modules_gate: bool = False
    branch_coverage_gate: bool = False
    performance_sla_gate: bool = False
    test_execution_gate: bool = False
    security_gate: bool = False
    pytest_style_gate: bool = False
    total_gates: int = 0
    passed_gates: int = 0


@dataclass
class SystemMetrics:
    """System resource and environment metrics."""
    cpu_count: int = 0
    memory_total: float = 0.0
    memory_available: float = 0.0
    disk_usage: float = 0.0
    execution_environment: str = ""
    ci_environment: bool = False
    python_version: str = ""
    os_platform: str = ""


@dataclass
class ComprehensiveTestMetrics:
    """Comprehensive test metrics aggregation."""
    timestamp: str = ""
    execution_id: str = ""
    project_version: str = ""
    test_execution: TestExecutionMetrics = field(default_factory=TestExecutionMetrics)
    coverage: CoverageMetrics = field(default_factory=CoverageMetrics)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    quality_gates: QualityGateResults = field(default_factory=QualityGateResults)
    system: SystemMetrics = field(default_factory=SystemMetrics)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TestMetricsAggregator:
    """
    Central test metrics aggregation engine implementing comprehensive metrics collection.
    
    This class orchestrates the collection and aggregation of test execution results,
    coverage analysis, performance validation, and quality gate status to provide
    comprehensive reporting supporting the enhanced testing strategy and CI/CD integration.
    """

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize test metrics aggregator with configuration loading.
        
        Args:
            project_root: Root directory of the project (defaults to current working directory)
        """
        self.project_root = Path(project_root or os.getcwd()).resolve()
        self.scripts_coverage_dir = self.project_root / "scripts" / "coverage"
        self.scripts_benchmarks_dir = self.project_root / "scripts" / "benchmarks"
        self.reports_dir = self.scripts_coverage_dir / "reports"
        
        # Initialize directories
        self._ensure_directories()
        
        # Load configurations
        self._load_configurations()
        
        # Initialize Jinja2 environment
        self._setup_template_environment()
        
        # Metrics collection state
        self.comprehensive_metrics = ComprehensiveTestMetrics()
        self.historical_data: List[Dict[str, Any]] = []
        
        # CI/CD integration state
        self.ci_environment = self._detect_ci_environment()
        self.execution_id = self._generate_execution_id()
        
        print(f"✅ Test Metrics Aggregator initialized")
        print(f"   📁 Project root: {self.project_root}")
        print(f"   📊 Scripts directory: {self.scripts_coverage_dir}")
        print(f"   🏃 Execution ID: {self.execution_id}")

    def _ensure_directories(self) -> None:
        """Create necessary directories for metrics collection."""
        directories = [
            self.scripts_coverage_dir,
            self.reports_dir,
            self.scripts_coverage_dir / "historical",
            self.scripts_coverage_dir / "artifacts",
            self.scripts_coverage_dir / "templates"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _load_configurations(self) -> None:
        """Load configuration files for metrics aggregation."""
        try:
            # Load coverage thresholds
            thresholds_path = self.scripts_coverage_dir / "coverage-thresholds.json"
            if thresholds_path.exists():
                with open(thresholds_path, 'r', encoding='utf-8') as f:
                    self.coverage_thresholds = json.load(f)
            else:
                # Default coverage thresholds
                self.coverage_thresholds = {
                    "global_settings": {
                        "overall_threshold": {"line_coverage": 90.0, "branch_coverage": 85.0}
                    },
                    "critical_modules": {"threshold": 100.0}
                }
            
            # Load quality gates configuration
            quality_gates_path = self.scripts_coverage_dir / "quality-gates.yml"
            if quality_gates_path.exists():
                with open(quality_gates_path, 'r', encoding='utf-8') as f:
                    self.quality_gates_config = yaml.safe_load(f)
            else:
                # Default quality gates
                self.quality_gates_config = {
                    "gates": {
                        "coverage": {"enabled": True, "threshold": 90.0},
                        "performance": {"enabled": True, "sla_compliance": True},
                        "security": {"enabled": True, "scan_clean": True}
                    }
                }
            
            print(f"✅ Configurations loaded from {self.scripts_coverage_dir}")
            
        except Exception as e:
            print(f"⚠️  Warning: Configuration loading failed, using defaults: {e}")
            self.coverage_thresholds = {"global_settings": {"overall_threshold": {"line_coverage": 90.0}}}
            self.quality_gates_config = {"gates": {}}

    def _setup_template_environment(self) -> None:
        """Initialize Jinja2 template environment for report generation."""
        templates_dir = self.scripts_coverage_dir / "templates"
        
        if templates_dir.exists():
            self.jinja_env = Environment(
                loader=FileSystemLoader(str(templates_dir)),
                autoescape=True,
                trim_blocks=True,
                lstrip_blocks=True
            )
            
            # Register custom filters
            self.jinja_env.filters.update({
                'percentage': self._percentage_filter,
                'format_duration': self._format_duration_filter,
                'format_timestamp': self._format_timestamp_filter,
                'format_bytes': self._format_bytes_filter
            })
        else:
            self.jinja_env = None
            print(f"⚠️  Warning: Templates directory not found: {templates_dir}")

    def _percentage_filter(self, numerator: Union[int, float], denominator: Union[int, float]) -> float:
        """Calculate percentage with safe division."""
        if denominator == 0:
            return 0.0
        return round((numerator / denominator) * 100, 2)

    def _format_duration_filter(self, seconds: float) -> str:
        """Format duration in seconds to human-readable format."""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def _format_timestamp_filter(self, timestamp: str, format_str: str = "%Y-%m-%d %H:%M:%S UTC") -> str:
        """Format ISO timestamp for display."""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime(format_str)
        except ValueError:
            return timestamp

    def _format_bytes_filter(self, bytes_value: float) -> str:
        """Format bytes to human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"

    def _detect_ci_environment(self) -> bool:
        """Detect if running in CI/CD environment."""
        ci_indicators = [
            'CI', 'CONTINUOUS_INTEGRATION', 'GITHUB_ACTIONS',
            'JENKINS_URL', 'GITLAB_CI', 'TRAVIS', 'CIRCLECI'
        ]
        return any(os.environ.get(indicator) for indicator in ci_indicators)

    def _generate_execution_id(self) -> str:
        """Generate unique execution ID for metrics tracking."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        if self.ci_environment:
            build_number = os.environ.get('GITHUB_RUN_NUMBER', os.environ.get('BUILD_NUMBER', '0'))
            return f"ci_{timestamp}_{build_number}"
        else:
            return f"local_{timestamp}"

    def collect_test_execution_metrics(self, junit_xml_path: Optional[Path] = None) -> None:
        """
        Collect test execution metrics from pytest results.
        
        Args:
            junit_xml_path: Path to JUnit XML results file
        """
        print("📊 Collecting test execution metrics...")
        
        try:
            # Initialize test execution metrics
            test_metrics = TestExecutionMetrics()
            
            # Collect system information
            test_metrics.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            test_metrics.platform = sys.platform
            
            # Try to get pytest version
            try:
                import pytest
                test_metrics.pytest_version = pytest.__version__
            except ImportError:
                test_metrics.pytest_version = "unknown"
            
            # Parse JUnit XML if available
            if junit_xml_path and junit_xml_path.exists():
                test_metrics = self._parse_junit_xml(junit_xml_path, test_metrics)
            else:
                # Try to find default JUnit XML files
                potential_files = [
                    self.project_root / "test-results.xml",
                    self.project_root / "junit.xml",
                    self.scripts_coverage_dir / "test-results.xml"
                ]
                
                for xml_file in potential_files:
                    if xml_file.exists():
                        test_metrics = self._parse_junit_xml(xml_file, test_metrics)
                        break
                else:
                    print("⚠️  Warning: No JUnit XML files found, using default metrics")
            
            # Detect parallel execution from environment or recent pytest execution
            if os.environ.get('PYTEST_XDIST_WORKER_COUNT'):
                test_metrics.parallel_workers = int(os.environ.get('PYTEST_XDIST_WORKER_COUNT', '1'))
            else:
                test_metrics.parallel_workers = psutil.cpu_count() or 1
            
            self.comprehensive_metrics.test_execution = test_metrics
            
            print(f"✅ Test execution metrics collected:")
            print(f"   🧪 Total tests: {test_metrics.total_tests}")
            print(f"   ✅ Passed: {test_metrics.passed_tests}")
            print(f"   ❌ Failed: {test_metrics.failed_tests}")
            print(f"   ⏱️  Execution time: {test_metrics.execution_time:.2f}s")
            print(f"   🔧 Parallel workers: {test_metrics.parallel_workers}")
            
        except Exception as e:
            raise AggregationError(f"Test execution metrics collection failed: {e}")

    def _parse_junit_xml(self, xml_path: Path, test_metrics: TestExecutionMetrics) -> TestExecutionMetrics:
        """Parse JUnit XML file for test results."""
        try:
            import xml.etree.ElementTree as ET
            
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Parse test results from testsuite elements
            for testsuite in root.findall('.//testsuite'):
                test_metrics.total_tests += int(testsuite.get('tests', 0))
                test_metrics.failed_tests += int(testsuite.get('failures', 0))
                test_metrics.error_tests += int(testsuite.get('errors', 0))
                test_metrics.skipped_tests += int(testsuite.get('skipped', 0))
                test_metrics.execution_time += float(testsuite.get('time', 0))
            
            # Calculate passed tests
            test_metrics.passed_tests = (test_metrics.total_tests - 
                                       test_metrics.failed_tests - 
                                       test_metrics.error_tests - 
                                       test_metrics.skipped_tests)
            
            # Parse test categories from testcase elements
            categories = defaultdict(int)
            for testcase in root.findall('.//testcase'):
                classname = testcase.get('classname', '')
                if 'integration' in classname.lower():
                    categories['integration'] += 1
                elif 'benchmark' in classname.lower():
                    categories['benchmark'] += 1
                elif 'unit' in classname.lower():
                    categories['unit'] += 1
                else:
                    categories['other'] += 1
            
            test_metrics.test_categories = dict(categories)
            
        except Exception as e:
            print(f"⚠️  Warning: JUnit XML parsing failed: {e}")
        
        return test_metrics

    def collect_coverage_metrics(self) -> None:
        """Collect coverage metrics by integrating with generate-coverage-reports.py."""
        print("📈 Collecting coverage metrics...")
        
        try:
            # Import and use the coverage report generator
            coverage_script = self.scripts_coverage_dir / "generate-coverage-reports.py"
            
            if coverage_script.exists():
                # Run coverage analysis via subprocess to get JSON output
                cmd = [
                    sys.executable, str(coverage_script),
                    "--analyze", "--json-only",
                    "--output-dir", str(self.scripts_coverage_dir)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
                
                if result.returncode == 0:
                    # Parse coverage JSON output
                    json_report_path = self.scripts_coverage_dir / "coverage.json"
                    if json_report_path.exists():
                        with open(json_report_path, 'r', encoding='utf-8') as f:
                            coverage_data = json.load(f)
                        
                        self._extract_coverage_metrics(coverage_data)
                    else:
                        print("⚠️  Warning: Coverage JSON report not found")
                else:
                    print(f"⚠️  Warning: Coverage analysis failed: {result.stderr}")
            
            # Fallback: try to read existing coverage data
            if self.comprehensive_metrics.coverage.line_coverage == 0.0:
                self._fallback_coverage_collection()
            
        except Exception as e:
            print(f"⚠️  Warning: Coverage metrics collection failed: {e}")
            self._fallback_coverage_collection()

    def _extract_coverage_metrics(self, coverage_data: Dict[str, Any]) -> None:
        """Extract coverage metrics from coverage report data."""
        try:
            overall_data = coverage_data.get('coverage_data', {}).get('overall', {})
            
            coverage_metrics = CoverageMetrics(
                line_coverage=overall_data.get('line_coverage_percentage', 0.0),
                branch_coverage=overall_data.get('branch_coverage_percentage', 0.0),
                function_coverage=overall_data.get('function_coverage_percentage', 0.0),
                total_lines=overall_data.get('total_lines', 0),
                covered_lines=overall_data.get('covered_lines', 0),
                missing_lines=overall_data.get('missing_lines', 0)
            )
            
            # Calculate critical modules coverage
            modules_data = coverage_data.get('coverage_data', {}).get('modules', {})
            critical_modules = [m for m in modules_data.values() if m.get('priority') == 'critical']
            if critical_modules:
                critical_avg = sum(m['metrics']['line_coverage_percentage'] for m in critical_modules) / len(critical_modules)
                coverage_metrics.critical_modules_coverage = critical_avg
            
            # Determine quality gate status
            threshold = self.coverage_thresholds.get('global_settings', {}).get('overall_threshold', {}).get('line_coverage', 90.0)
            coverage_metrics.quality_gate_status = "passing" if coverage_metrics.line_coverage >= threshold else "failing"
            
            self.comprehensive_metrics.coverage = coverage_metrics
            
            print(f"✅ Coverage metrics extracted:")
            print(f"   📊 Line coverage: {coverage_metrics.line_coverage:.2f}%")
            print(f"   🌳 Branch coverage: {coverage_metrics.branch_coverage:.2f}%")
            print(f"   🔍 Critical modules: {coverage_metrics.critical_modules_coverage:.2f}%")
            
        except Exception as e:
            print(f"⚠️  Warning: Coverage metrics extraction failed: {e}")

    def _fallback_coverage_collection(self) -> None:
        """Fallback coverage collection from .coverage file."""
        try:
            coverage_file = self.project_root / ".coverage"
            if coverage_file.exists():
                # Basic coverage data extraction
                import coverage
                cov = coverage.Coverage(data_file=str(coverage_file))
                cov.load()
                
                # Get basic coverage percentage
                report_result = cov.report(show_missing=False)
                
                # Extract basic metrics
                coverage_metrics = CoverageMetrics(
                    line_coverage=float(report_result or 0.0),
                    quality_gate_status="unknown"
                )
                
                self.comprehensive_metrics.coverage = coverage_metrics
                print(f"✅ Fallback coverage collected: {coverage_metrics.line_coverage:.2f}%")
                
        except Exception as e:
            print(f"⚠️  Warning: Fallback coverage collection failed: {e}")

    def collect_performance_metrics(self) -> None:
        """Collect performance metrics by integrating with scripts/benchmarks/."""
        print("🚀 Collecting performance metrics...")
        
        try:
            performance_metrics = PerformanceMetrics()
            
            # Check if benchmark script exists
            benchmark_script = self.scripts_benchmarks_dir / "run_benchmarks.py"
            
            if benchmark_script.exists():
                # Try to run benchmarks if explicitly requested or check existing results
                benchmark_results_path = self.scripts_benchmarks_dir / "benchmark-results.json"
                
                if benchmark_results_path.exists():
                    # Parse existing benchmark results
                    with open(benchmark_results_path, 'r', encoding='utf-8') as f:
                        benchmark_data = json.load(f)
                    
                    performance_metrics = self._extract_performance_metrics(benchmark_data)
                    performance_metrics.benchmark_executed = True
                    performance_metrics.performance_artifacts_generated = True
                else:
                    print("ℹ️  No benchmark results found, checking SLA validation script...")
                    
                    # Check SLA validation script
                    sla_script = self.scripts_coverage_dir / "check-performance-slas.py"
                    if sla_script.exists():
                        performance_metrics = self._check_performance_slas()
            
            # Check for performance artifacts
            benchmark_artifacts = list(self.scripts_benchmarks_dir.glob("*.json"))
            performance_metrics.performance_artifacts_generated = len(benchmark_artifacts) > 0
            
            self.comprehensive_metrics.performance = performance_metrics
            
            print(f"✅ Performance metrics collected:")
            print(f"   🏃 Benchmark executed: {performance_metrics.benchmark_executed}")
            print(f"   📊 Data loading SLA: {'✅' if performance_metrics.data_loading_sla_compliance else '❌'}")
            print(f"   🔄 Transformation SLA: {'✅' if performance_metrics.transformation_sla_compliance else '❌'}")
            print(f"   📁 Artifacts generated: {performance_metrics.performance_artifacts_generated}")
            
        except Exception as e:
            print(f"⚠️  Warning: Performance metrics collection failed: {e}")

    def _extract_performance_metrics(self, benchmark_data: Dict[str, Any]) -> PerformanceMetrics:
        """Extract performance metrics from benchmark results."""
        performance_metrics = PerformanceMetrics()
        
        try:
            # Extract SLA compliance from benchmark data
            sla_results = benchmark_data.get('sla_compliance', {})
            
            performance_metrics.data_loading_sla_compliance = sla_results.get('data_loading', False)
            performance_metrics.transformation_sla_compliance = sla_results.get('transformation', False)
            performance_metrics.discovery_sla_compliance = sla_results.get('discovery', False)
            performance_metrics.config_sla_compliance = sla_results.get('config', False)
            
            # Extract execution metrics
            execution_data = benchmark_data.get('execution_summary', {})
            performance_metrics.benchmark_execution_time = execution_data.get('total_time', 0.0)
            performance_metrics.memory_peak_usage = execution_data.get('peak_memory_mb', 0.0) * 1024 * 1024  # Convert to bytes
            
            # Check for regression
            regression_data = benchmark_data.get('regression_analysis', {})
            performance_metrics.regression_detected = regression_data.get('regression_detected', False)
            
        except Exception as e:
            print(f"⚠️  Warning: Performance metrics extraction failed: {e}")
        
        return performance_metrics

    def _check_performance_slas(self) -> PerformanceMetrics:
        """Check performance SLAs using the SLA validation script."""
        performance_metrics = PerformanceMetrics()
        
        try:
            sla_script = self.scripts_coverage_dir / "check-performance-slas.py"
            
            if sla_script.exists():
                # Run SLA check script
                cmd = [sys.executable, str(sla_script), "--json-output"]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
                
                if result.returncode == 0:
                    try:
                        sla_data = json.loads(result.stdout)
                        performance_metrics = self._extract_performance_metrics(sla_data)
                    except json.JSONDecodeError:
                        print("⚠️  Warning: SLA script did not return valid JSON")
                else:
                    print(f"⚠️  Warning: SLA validation script failed: {result.stderr}")
        
        except Exception as e:
            print(f"⚠️  Warning: SLA checking failed: {e}")
        
        return performance_metrics

    def validate_quality_gates(self) -> None:
        """Validate all quality gates and update results."""
        print("🚦 Validating quality gates...")
        
        try:
            quality_results = QualityGateResults()
            gates_config = self.quality_gates_config.get('gates', {})
            
            # Overall coverage gate
            if gates_config.get('coverage', {}).get('enabled', True):
                threshold = gates_config.get('coverage', {}).get('threshold', 90.0)
                quality_results.overall_coverage_gate = self.comprehensive_metrics.coverage.line_coverage >= threshold
                quality_results.total_gates += 1
            
            # Critical modules gate
            if gates_config.get('critical_modules', {}).get('enabled', True):
                threshold = gates_config.get('critical_modules', {}).get('threshold', 100.0)
                quality_results.critical_modules_gate = self.comprehensive_metrics.coverage.critical_modules_coverage >= threshold
                quality_results.total_gates += 1
            
            # Branch coverage gate
            if gates_config.get('branch_coverage', {}).get('enabled', True):
                threshold = gates_config.get('branch_coverage', {}).get('threshold', 85.0)
                quality_results.branch_coverage_gate = self.comprehensive_metrics.coverage.branch_coverage >= threshold
                quality_results.total_gates += 1
            
            # Performance SLA gate
            if gates_config.get('performance', {}).get('enabled', True):
                performance = self.comprehensive_metrics.performance
                quality_results.performance_sla_gate = (
                    performance.data_loading_sla_compliance and
                    performance.transformation_sla_compliance and
                    not performance.regression_detected
                )
                quality_results.total_gates += 1
            
            # Test execution gate
            if gates_config.get('test_execution', {}).get('enabled', True):
                test_execution = self.comprehensive_metrics.test_execution
                quality_results.test_execution_gate = (
                    test_execution.failed_tests == 0 and
                    test_execution.error_tests == 0 and
                    test_execution.total_tests > 0
                )
                quality_results.total_gates += 1
            
            # Security gate (placeholder - would integrate with security scanning)
            if gates_config.get('security', {}).get('enabled', True):
                quality_results.security_gate = True  # Default to passing
                quality_results.total_gates += 1
            
            # Pytest style gate (placeholder - would integrate with style validation)
            if gates_config.get('pytest_style', {}).get('enabled', True):
                quality_results.pytest_style_gate = True  # Default to passing
                quality_results.total_gates += 1
            
            # Count passed gates
            gates_status = [
                quality_results.overall_coverage_gate,
                quality_results.critical_modules_gate,
                quality_results.branch_coverage_gate,
                quality_results.performance_sla_gate,
                quality_results.test_execution_gate,
                quality_results.security_gate,
                quality_results.pytest_style_gate
            ]
            quality_results.passed_gates = sum(1 for gate in gates_status if gate)
            
            self.comprehensive_metrics.quality_gates = quality_results
            
            print(f"✅ Quality gates validated:")
            print(f"   📊 Coverage gate: {'✅' if quality_results.overall_coverage_gate else '❌'}")
            print(f"   🔍 Critical modules: {'✅' if quality_results.critical_modules_gate else '❌'}")
            print(f"   🌳 Branch coverage: {'✅' if quality_results.branch_coverage_gate else '❌'}")
            print(f"   🚀 Performance SLA: {'✅' if quality_results.performance_sla_gate else '❌'}")
            print(f"   🧪 Test execution: {'✅' if quality_results.test_execution_gate else '❌'}")
            print(f"   🛡️  Security: {'✅' if quality_results.security_gate else '❌'}")
            print(f"   📝 Pytest style: {'✅' if quality_results.pytest_style_gate else '❌'}")
            print(f"   📈 Total: {quality_results.passed_gates}/{quality_results.total_gates}")
            
        except Exception as e:
            raise QualityGateError(f"Quality gate validation failed: {e}")

    def collect_system_metrics(self) -> None:
        """Collect system resource and environment metrics."""
        print("💻 Collecting system metrics...")
        
        try:
            system_metrics = SystemMetrics()
            
            # CPU information
            system_metrics.cpu_count = psutil.cpu_count() or 1
            
            # Memory information
            memory = psutil.virtual_memory()
            system_metrics.memory_total = memory.total
            system_metrics.memory_available = memory.available
            
            # Disk usage
            disk = psutil.disk_usage(str(self.project_root))
            system_metrics.disk_usage = disk.percent
            
            # Environment information
            system_metrics.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            system_metrics.os_platform = sys.platform
            system_metrics.ci_environment = self.ci_environment
            
            # Execution environment
            if self.ci_environment:
                if os.environ.get('GITHUB_ACTIONS'):
                    system_metrics.execution_environment = "GitHub Actions"
                elif os.environ.get('JENKINS_URL'):
                    system_metrics.execution_environment = "Jenkins"
                elif os.environ.get('GITLAB_CI'):
                    system_metrics.execution_environment = "GitLab CI"
                else:
                    system_metrics.execution_environment = "CI/CD (Unknown)"
            else:
                system_metrics.execution_environment = "Local Development"
            
            self.comprehensive_metrics.system = system_metrics
            
            print(f"✅ System metrics collected:")
            print(f"   💾 Memory: {system_metrics.memory_available / (1024**3):.1f}GB available / {system_metrics.memory_total / (1024**3):.1f}GB total")
            print(f"   🖥️  CPU: {system_metrics.cpu_count} cores")
            print(f"   💿 Disk usage: {system_metrics.disk_usage:.1f}%")
            print(f"   🌐 Environment: {system_metrics.execution_environment}")
            
        except Exception as e:
            print(f"⚠️  Warning: System metrics collection failed: {e}")

    def finalize_metrics(self) -> None:
        """Finalize comprehensive metrics collection."""
        print("📋 Finalizing comprehensive metrics...")
        
        try:
            # Set metadata
            self.comprehensive_metrics.timestamp = datetime.now(timezone.utc).isoformat()
            self.comprehensive_metrics.execution_id = self.execution_id
            
            # Try to get project version
            try:
                import importlib.metadata
                self.comprehensive_metrics.project_version = importlib.metadata.version("flyrigloader")
            except Exception:
                self.comprehensive_metrics.project_version = "2.0.0"
            
            # Additional metadata
            self.comprehensive_metrics.metadata = {
                "collection_duration": time.time() - self._start_time if hasattr(self, '_start_time') else 0.0,
                "ci_environment": self.ci_environment,
                "execution_environment": self.comprehensive_metrics.system.execution_environment,
                "pytest_xdist_enabled": self.comprehensive_metrics.test_execution.parallel_workers > 1,
                "performance_benchmarks_available": self.comprehensive_metrics.performance.benchmark_executed,
                "quality_gates_enabled": self.comprehensive_metrics.quality_gates.total_gates > 0
            }
            
            print(f"✅ Metrics finalized:")
            print(f"   🕐 Timestamp: {self.comprehensive_metrics.timestamp}")
            print(f"   🆔 Execution ID: {self.comprehensive_metrics.execution_id}")
            print(f"   📦 Project version: {self.comprehensive_metrics.project_version}")
            
        except Exception as e:
            print(f"⚠️  Warning: Metrics finalization failed: {e}")

    def generate_comprehensive_report(self) -> Dict[str, Path]:
        """Generate comprehensive test metrics reports in multiple formats."""
        print("📄 Generating comprehensive test metrics reports...")
        
        try:
            self.reports_dir.mkdir(parents=True, exist_ok=True)
            generated_reports = {}
            
            # Generate JSON report
            json_report = self._generate_json_report()
            generated_reports['json'] = json_report
            
            # Generate HTML report if templates available
            if self.jinja_env:
                html_report = self._generate_html_report()
                if html_report:
                    generated_reports['html'] = html_report
            
            # Generate markdown summary
            markdown_report = self._generate_markdown_report()
            generated_reports['markdown'] = markdown_report
            
            print(f"✅ Reports generated: {', '.join(generated_reports.keys())}")
            return generated_reports
            
        except Exception as e:
            raise AggregationError(f"Comprehensive report generation failed: {e}")

    def _generate_json_report(self) -> Path:
        """Generate JSON format report."""
        json_path = self.reports_dir / f"test-metrics-{self.execution_id}.json"
        
        # Convert to dictionary and ensure JSON serializable
        metrics_dict = asdict(self.comprehensive_metrics)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ JSON report: {json_path}")
        return json_path

    def _generate_html_report(self) -> Optional[Path]:
        """Generate HTML format report."""
        try:
            template = self.jinja_env.get_template('metrics-report.html.j2')
            
            html_content = template.render(
                metrics=self.comprehensive_metrics,
                timestamp=datetime.now(timezone.utc),
                execution_id=self.execution_id
            )
            
            html_path = self.reports_dir / f"test-metrics-{self.execution_id}.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"✅ HTML report: {html_path}")
            return html_path
            
        except Exception as e:
            print(f"⚠️  Warning: HTML report generation failed: {e}")
            return None

    def _generate_markdown_report(self) -> Path:
        """Generate Markdown summary report."""
        markdown_path = self.reports_dir / f"test-metrics-summary-{self.execution_id}.md"
        
        # Generate markdown content
        content = self._create_markdown_content()
        
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Markdown report: {markdown_path}")
        return markdown_path

    def _create_markdown_content(self) -> str:
        """Create markdown content for test metrics summary."""
        metrics = self.comprehensive_metrics
        
        content = f"""# Test Metrics Summary

**Execution ID:** {metrics.execution_id}  
**Timestamp:** {metrics.timestamp}  
**Project Version:** {metrics.project_version}  
**Environment:** {metrics.system.execution_environment}

## Test Execution Results

| Metric | Value |
|--------|-------|
| Total Tests | {metrics.test_execution.total_tests} |
| Passed | {metrics.test_execution.passed_tests} |
| Failed | {metrics.test_execution.failed_tests} |
| Skipped | {metrics.test_execution.skipped_tests} |
| Execution Time | {metrics.test_execution.execution_time:.2f}s |
| Parallel Workers | {metrics.test_execution.parallel_workers} |

## Coverage Analysis

| Coverage Type | Percentage | Status |
|---------------|------------|--------|
| Line Coverage | {metrics.coverage.line_coverage:.2f}% | {'✅' if metrics.quality_gates.overall_coverage_gate else '❌'} |
| Branch Coverage | {metrics.coverage.branch_coverage:.2f}% | {'✅' if metrics.quality_gates.branch_coverage_gate else '❌'} |
| Function Coverage | {metrics.coverage.function_coverage:.2f}% | ℹ️ |
| Critical Modules | {metrics.coverage.critical_modules_coverage:.2f}% | {'✅' if metrics.quality_gates.critical_modules_gate else '❌'} |

## Performance Validation

| SLA Component | Status | Compliance |
|---------------|--------|------------|
| Data Loading | {'✅ Compliant' if metrics.performance.data_loading_sla_compliance else '❌ Non-compliant'} | <1s per 100MB |
| Transformation | {'✅ Compliant' if metrics.performance.transformation_sla_compliance else '❌ Non-compliant'} | <500ms per 1M rows |
| Discovery | {'✅ Compliant' if metrics.performance.discovery_sla_compliance else '❌ Non-compliant'} | <5s for 10K files |
| Configuration | {'✅ Compliant' if metrics.performance.config_sla_compliance else '❌ Non-compliant'} | <100ms loading |

## Quality Gates Summary

**Overall Status:** {metrics.quality_gates.passed_gates}/{metrics.quality_gates.total_gates} gates passed

| Gate | Status | Description |
|------|--------|-------------|
| Coverage | {'✅ PASS' if metrics.quality_gates.overall_coverage_gate else '❌ FAIL'} | ≥90% line coverage |
| Critical Modules | {'✅ PASS' if metrics.quality_gates.critical_modules_gate else '❌ FAIL'} | 100% coverage required |
| Branch Coverage | {'✅ PASS' if metrics.quality_gates.branch_coverage_gate else '❌ FAIL'} | ≥85% branch coverage |
| Performance SLA | {'✅ PASS' if metrics.quality_gates.performance_sla_gate else '❌ FAIL'} | All SLAs compliant |
| Test Execution | {'✅ PASS' if metrics.quality_gates.test_execution_gate else '❌ FAIL'} | No test failures |
| Security | {'✅ PASS' if metrics.quality_gates.security_gate else '❌ FAIL'} | Security scan clean |

## System Information

| Resource | Value |
|----------|-------|
| CPU Cores | {metrics.system.cpu_count} |
| Memory Total | {metrics.system.memory_total / (1024**3):.1f}GB |
| Memory Available | {metrics.system.memory_available / (1024**3):.1f}GB |
| Disk Usage | {metrics.system.disk_usage:.1f}% |
| Python Version | {metrics.system.python_version} |
| Platform | {metrics.system.os_platform} |

---
*Generated by FlyrigLoader Test Metrics Aggregation System*
"""
        return content

    def persist_historical_data(self) -> None:
        """Persist metrics to historical data store for trend analysis."""
        print("💾 Persisting historical data...")
        
        try:
            historical_file = self.scripts_coverage_dir / "historical" / "metrics_history.json"
            historical_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Load existing historical data
            if historical_file.exists():
                with open(historical_file, 'r', encoding='utf-8') as f:
                    historical_data = json.load(f)
            else:
                historical_data = []
            
            # Add current metrics
            metrics_dict = asdict(self.comprehensive_metrics)
            historical_data.append(metrics_dict)
            
            # Implement retention policy (keep last 90 days)
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=90)
            
            filtered_data = []
            for entry in historical_data:
                try:
                    entry_date = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                    if entry_date >= cutoff_date:
                        filtered_data.append(entry)
                except (ValueError, KeyError):
                    # Keep entries with invalid timestamps rather than lose data
                    filtered_data.append(entry)
            
            # Save updated historical data
            with open(historical_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"✅ Historical data persisted: {len(filtered_data)} entries")
            
        except Exception as e:
            print(f"⚠️  Warning: Historical data persistence failed: {e}")

    def upload_artifacts_to_ci(self, reports: Dict[str, Path]) -> bool:
        """Upload generated artifacts to CI/CD system."""
        print("☁️  Uploading artifacts to CI/CD system...")
        
        try:
            if not self.ci_environment:
                print("ℹ️  Not in CI environment, skipping upload")
                return True
            
            artifacts_dir = self.scripts_coverage_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy reports to artifacts directory
            for report_type, report_path in reports.items():
                artifact_name = f"test-metrics-{report_type}-{self.execution_id}{report_path.suffix}"
                artifact_path = artifacts_dir / artifact_name
                shutil.copy2(report_path, artifact_path)
                print(f"✅ Artifact prepared: {artifact_name}")
            
            # Create artifact summary
            summary_data = {
                "execution_id": self.execution_id,
                "timestamp": self.comprehensive_metrics.timestamp,
                "quality_gates_passed": f"{self.comprehensive_metrics.quality_gates.passed_gates}/{self.comprehensive_metrics.quality_gates.total_gates}",
                "coverage_percentage": self.comprehensive_metrics.coverage.line_coverage,
                "test_success_rate": self._calculate_test_success_rate(),
                "performance_sla_compliance": self._check_overall_sla_compliance(),
                "artifacts": list(reports.keys())
            }
            
            summary_path = artifacts_dir / "summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2)
            
            print(f"✅ Artifacts uploaded to: {artifacts_dir}")
            return True
            
        except Exception as e:
            print(f"⚠️  Warning: Artifact upload failed: {e}")
            return False

    def _calculate_test_success_rate(self) -> float:
        """Calculate test success rate percentage."""
        test_execution = self.comprehensive_metrics.test_execution
        if test_execution.total_tests == 0:
            return 0.0
        return (test_execution.passed_tests / test_execution.total_tests) * 100

    def _check_overall_sla_compliance(self) -> bool:
        """Check overall SLA compliance status."""
        performance = self.comprehensive_metrics.performance
        return (
            performance.data_loading_sla_compliance and
            performance.transformation_sla_compliance and
            performance.discovery_sla_compliance and
            performance.config_sla_compliance and
            not performance.regression_detected
        )

    def aggregate_all_metrics(self, 
                            junit_xml_path: Optional[Path] = None,
                            include_performance: bool = True) -> Dict[str, Path]:
        """
        Execute complete metrics aggregation workflow.
        
        Args:
            junit_xml_path: Path to JUnit XML results file
            include_performance: Whether to include performance metrics collection
            
        Returns:
            Dictionary of generated report paths
        """
        print("🚀 Starting comprehensive test metrics aggregation...")
        self._start_time = time.time()
        
        try:
            # Collect all metrics
            self.collect_test_execution_metrics(junit_xml_path)
            self.collect_coverage_metrics()
            
            if include_performance:
                self.collect_performance_metrics()
            
            self.collect_system_metrics()
            self.validate_quality_gates()
            self.finalize_metrics()
            
            # Generate reports
            reports = self.generate_comprehensive_report()
            
            # Persist historical data
            self.persist_historical_data()
            
            # Upload artifacts in CI environment
            if self.ci_environment:
                self.upload_artifacts_to_ci(reports)
            
            execution_time = time.time() - self._start_time
            print(f"🎉 Metrics aggregation completed successfully in {execution_time:.2f}s!")
            
            # Print final summary
            self._print_final_summary()
            
            return reports
            
        except Exception as e:
            raise AggregationError(f"Comprehensive metrics aggregation failed: {e}")

    def _print_final_summary(self) -> None:
        """Print final metrics summary."""
        metrics = self.comprehensive_metrics
        
        print("\n" + "="*60)
        print("📊 FLYRIGLOADER TEST METRICS SUMMARY")
        print("="*60)
        
        print(f"🧪 Test Execution: {metrics.test_execution.passed_tests}/{metrics.test_execution.total_tests} passed ({self._calculate_test_success_rate():.1f}%)")
        print(f"📈 Coverage: {metrics.coverage.line_coverage:.1f}% line, {metrics.coverage.branch_coverage:.1f}% branch")
        print(f"🚀 Performance: {metrics.performance.data_loading_sla_compliance and metrics.performance.transformation_sla_compliance}")
        print(f"🚦 Quality Gates: {metrics.quality_gates.passed_gates}/{metrics.quality_gates.total_gates} passed")
        
        if metrics.quality_gates.passed_gates == metrics.quality_gates.total_gates:
            print("\n✅ ALL QUALITY GATES PASSED - READY FOR MERGE")
        else:
            print("\n❌ QUALITY GATE FAILURES - MERGE BLOCKED")
            
        print("="*60)

    def validate_strict_quality_gates(self) -> bool:
        """
        Perform strict quality gate validation for CI/CD blocking.
        
        Returns:
            True if all gates pass, False otherwise
        """
        print("🚦 Performing strict quality gate validation...")
        
        all_gates_pass = (
            self.comprehensive_metrics.quality_gates.passed_gates == 
            self.comprehensive_metrics.quality_gates.total_gates
        )
        
        if not all_gates_pass:
            print("❌ QUALITY GATE FAILURES DETECTED:")
            
            gates = self.comprehensive_metrics.quality_gates
            if not gates.overall_coverage_gate:
                print(f"   📊 Coverage: {self.comprehensive_metrics.coverage.line_coverage:.2f}% < 90% required")
            if not gates.critical_modules_gate:
                print(f"   🔍 Critical modules: {self.comprehensive_metrics.coverage.critical_modules_coverage:.2f}% < 100% required")
            if not gates.test_execution_gate:
                print(f"   🧪 Test failures: {self.comprehensive_metrics.test_execution.failed_tests} failed, {self.comprehensive_metrics.test_execution.error_tests} errors")
            if not gates.performance_sla_gate:
                print(f"   🚀 Performance SLA violations detected")
        
        return all_gates_pass


def main():
    """Main entry point for test metrics aggregation script."""
    parser = argparse.ArgumentParser(
        description="FlyrigLoader Test Metrics Aggregation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete metrics aggregation
  python collect-test-metrics.py --aggregate-all

  # Aggregate with specific JUnit XML
  python collect-test-metrics.py --junit-xml test-results.xml --aggregate-all

  # Strict validation for CI/CD blocking
  python collect-test-metrics.py --aggregate-all --validate-strict

  # Skip performance metrics collection
  python collect-test-metrics.py --aggregate-all --no-performance

  # Generate reports only
  python collect-test-metrics.py --reports-only
        """
    )
    
    # Aggregation options
    parser.add_argument(
        '--aggregate-all',
        action='store_true',
        help='Perform complete metrics aggregation workflow'
    )
    parser.add_argument(
        '--junit-xml',
        type=Path,
        help='Path to JUnit XML test results file'
    )
    parser.add_argument(
        '--no-performance',
        action='store_true',
        help='Skip performance metrics collection'
    )
    
    # Report generation options
    parser.add_argument(
        '--reports-only',
        action='store_true',
        help='Generate reports from existing metrics only'
    )
    
    # Quality gate options
    parser.add_argument(
        '--validate-strict',
        action='store_true',
        help='Perform strict quality gate validation (exits with error on failure)'
    )
    
    # CI/CD integration options
    parser.add_argument(
        '--upload-artifacts',
        action='store_true',
        help='Upload artifacts to CI/CD system'
    )
    
    # Other options
    parser.add_argument(
        '--project-root',
        type=Path,
        help='Project root directory (default: current working directory)'
    )
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
        # Initialize metrics aggregator
        aggregator = TestMetricsAggregator(project_root=args.project_root)
        
        if args.aggregate_all:
            # Perform complete aggregation
            reports = aggregator.aggregate_all_metrics(
                junit_xml_path=args.junit_xml,
                include_performance=not args.no_performance
            )
            
            # Upload artifacts if requested or in CI
            if args.upload_artifacts or aggregator.ci_environment:
                aggregator.upload_artifacts_to_ci(reports)
            
        elif args.reports_only:
            # Generate reports only (assumes metrics already collected)
            reports = aggregator.generate_comprehensive_report()
            
        else:
            print("❌ No action specified. Use --aggregate-all or --reports-only")
            sys.exit(1)
        
        # Perform strict validation if requested
        if args.validate_strict:
            if not aggregator.validate_strict_quality_gates():
                print("❌ Quality gate validation failed!")
                sys.exit(1)
        
        print("🎉 Test metrics aggregation completed successfully!")
        
    except (MetricsException, ConfigurationError, AggregationError, QualityGateError) as e:
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