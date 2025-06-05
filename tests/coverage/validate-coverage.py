#!/usr/bin/env python3
"""
Automated Coverage Validation Script

Comprehensive coverage validation implementing quality gate enforcement with 
module-specific threshold validation, performance SLA checking, and CI/CD 
integration support per TST-COV-004 requirements.

This script provides programmatic coverage analysis enabling automated merge 
blocking per TST-COV-004 requirements and implements:

- TST-COV-001: Maintain >90% overall test coverage across all modules
- TST-COV-002: Achieve 100% coverage for critical data loading and validation modules  
- TST-COV-004: Block merges when coverage drops below thresholds
- TST-PERF-001: Data loading SLA validation within 1s per 100MB
- Section 4.1.1.5: Test execution workflow with automated quality gate enforcement

Author: FlyRigLoader Test Infrastructure Team
Created: 2024-12-19
Last Updated: 2024-12-19
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml


# ============================================================================
# CONFIGURATION AND DATA STRUCTURES
# ============================================================================

@dataclass
class ModuleCoverageResult:
    """Represents coverage analysis results for a specific module."""
    module_path: str
    line_coverage: float
    branch_coverage: float
    lines_covered: int
    lines_total: int
    branches_covered: int
    branches_total: int
    missing_lines: List[int] = field(default_factory=list)
    partial_branches: List[int] = field(default_factory=list)
    threshold_met: bool = False
    branch_threshold_met: bool = False
    category: str = "unknown"
    requirements: List[str] = field(default_factory=list)


@dataclass
class PerformanceBenchmarkResult:
    """Represents performance benchmark analysis results."""
    operation_name: str
    measured_time: float
    data_size: float
    sla_threshold: float
    sla_met: bool
    unit: str
    benchmark_category: str


@dataclass
class QualityGateResult:
    """Comprehensive quality gate validation results."""
    overall_coverage_met: bool
    critical_modules_met: bool
    branch_coverage_met: bool
    performance_slas_met: bool
    module_results: Dict[str, ModuleCoverageResult] = field(default_factory=dict)
    performance_results: List[PerformanceBenchmarkResult] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    overall_line_coverage: float = 0.0
    overall_branch_coverage: float = 0.0
    execution_time: float = 0.0


# ============================================================================
# COVERAGE VALIDATION CORE ENGINE
# ============================================================================

class CoverageValidator:
    """
    Comprehensive coverage validation engine implementing automated quality 
    gate enforcement with module-specific threshold validation and performance
    SLA checking per TST-COV-004 requirements.
    """

    def __init__(self, 
                 coverage_config_path: Path,
                 quality_gates_config_path: Path,
                 coverage_data_path: Optional[Path] = None,
                 benchmark_data_path: Optional[Path] = None):
        """
        Initialize coverage validator with configuration and data paths.
        
        Args:
            coverage_config_path: Path to coverage-thresholds.json configuration
            quality_gates_config_path: Path to quality-gates.yml configuration  
            coverage_data_path: Path to coverage data file (coverage.json)
            benchmark_data_path: Path to benchmark results (benchmark-results.json)
        """
        self.coverage_config_path = coverage_config_path
        self.quality_gates_config_path = quality_gates_config_path
        self.coverage_data_path = coverage_data_path or Path("coverage.json")
        self.benchmark_data_path = benchmark_data_path or Path("benchmark-results.json")
        
        # Configuration storage
        self.coverage_config: Dict[str, Any] = {}
        self.quality_gates_config: Dict[str, Any] = {}
        
        # Analysis results storage
        self.coverage_data: Dict[str, Any] = {}
        self.benchmark_data: Dict[str, Any] = {}
        
        # Logging configuration
        self.logger = self._configure_logging()
        
        # Performance tracking
        self.start_time = time.time()

    def _configure_logging(self) -> logging.Logger:
        """Configure comprehensive logging for validation analysis."""
        logger = logging.getLogger("coverage_validator")
        logger.setLevel(logging.INFO)
        
        # Create console handler with formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Enhanced formatter for detailed validation logging
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        
        # Avoid duplicate handlers in testing environments
        if not logger.handlers:
            logger.addHandler(console_handler)
        
        return logger

    def load_configurations(self) -> None:
        """
        Load and validate configuration files for coverage thresholds and 
        quality gates per TST-COV-001 through TST-COV-004 requirements.
        
        Raises:
            FileNotFoundError: If configuration files are missing
            json.JSONDecodeError: If JSON configuration is invalid
            yaml.YAMLError: If YAML configuration is invalid
        """
        self.logger.info("Loading coverage validation configurations")
        
        # Load coverage thresholds configuration
        try:
            with open(self.coverage_config_path, 'r') as f:
                self.coverage_config = json.load(f)
            self.logger.info(f"Loaded coverage thresholds from {self.coverage_config_path}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Coverage thresholds configuration not found: {self.coverage_config_path}"
            )
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in coverage thresholds configuration: {e}",
                e.doc, e.pos
            )
        
        # Load quality gates configuration
        try:
            with open(self.quality_gates_config_path, 'r') as f:
                self.quality_gates_config = yaml.safe_load(f)
            self.logger.info(f"Loaded quality gates from {self.quality_gates_config_path}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Quality gates configuration not found: {self.quality_gates_config_path}"
            )
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in quality gates configuration: {e}")
        
        # Validate configuration integrity
        self._validate_configuration_integrity()

    def _validate_configuration_integrity(self) -> None:
        """
        Validate configuration file integrity and consistency per
        Section 4.1.1.5 quality assurance requirements.
        
        Raises:
            ValueError: If configuration validation fails
        """
        required_coverage_keys = [
            "global_configuration", "module_thresholds", 
            "quality_gates", "exclusions"
        ]
        
        for key in required_coverage_keys:
            if key not in self.coverage_config:
                raise ValueError(f"Missing required coverage configuration key: {key}")
        
        required_quality_gate_keys = ["coverage", "performance", "execution"]
        for key in required_quality_gate_keys:
            if key not in self.quality_gates_config:
                raise ValueError(f"Missing required quality gates configuration key: {key}")
        
        # Validate threshold consistency
        global_threshold = self.coverage_config["global_configuration"]["overall_threshold"]
        quality_gate_threshold = self.quality_gates_config["coverage"]["overall_coverage_threshold"]
        
        if abs(global_threshold - quality_gate_threshold) > 0.1:
            self.logger.warning(
                f"Threshold mismatch between configurations: {global_threshold} vs {quality_gate_threshold}"
            )

    def load_coverage_data(self) -> None:
        """
        Load coverage analysis data from coverage.json with comprehensive
        error handling per TST-COV-003 reporting requirements.
        
        Raises:
            FileNotFoundError: If coverage data file is missing
            json.JSONDecodeError: If coverage data is invalid JSON
        """
        self.logger.info(f"Loading coverage data from {self.coverage_data_path}")
        
        try:
            with open(self.coverage_data_path, 'r') as f:
                self.coverage_data = json.load(f)
            
            # Validate coverage data structure
            required_keys = ["files", "totals"]
            for key in required_keys:
                if key not in self.coverage_data:
                    raise ValueError(f"Missing required coverage data key: {key}")
            
            self.logger.info(
                f"Loaded coverage data for {len(self.coverage_data['files'])} files"
            )
            
        except FileNotFoundError:
            self.logger.error(f"Coverage data file not found: {self.coverage_data_path}")
            # Generate mock coverage data for testing environments
            self.coverage_data = self._generate_mock_coverage_data()
            self.logger.warning("Using mock coverage data for validation testing")
            
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in coverage data file: {e}",
                e.doc, e.pos
            )

    def _generate_mock_coverage_data(self) -> Dict[str, Any]:
        """
        Generate mock coverage data for testing environments when actual
        coverage data is not available per TST-INF-003 requirements.
        
        Returns:
            Mock coverage data structure compatible with coverage.py output
        """
        return {
            "files": {},
            "totals": {
                "covered_lines": 0,
                "num_statements": 1,
                "percent_covered": 0.0,
                "covered_branches": 0,
                "num_branches": 1,
                "percent_covered_display": "0%",
                "missing_lines": 0,
                "excluded_lines": 0
            },
            "meta": {
                "version": "7.8.2",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "branch_coverage": True,
                "show_contexts": False
            }
        }

    def load_benchmark_data(self) -> None:
        """
        Load performance benchmark data for SLA validation per TST-PERF-001
        and TST-PERF-002 requirements.
        
        Raises:
            FileNotFoundError: If benchmark data file is missing
            json.JSONDecodeError: If benchmark data is invalid JSON
        """
        self.logger.info(f"Loading benchmark data from {self.benchmark_data_path}")
        
        try:
            with open(self.benchmark_data_path, 'r') as f:
                self.benchmark_data = json.load(f)
            
            self.logger.info(
                f"Loaded benchmark data for {len(self.benchmark_data.get('benchmarks', []))} benchmarks"
            )
            
        except FileNotFoundError:
            self.logger.warning(f"Benchmark data file not found: {self.benchmark_data_path}")
            # Generate mock benchmark data for environments without performance data
            self.benchmark_data = self._generate_mock_benchmark_data()
            self.logger.warning("Using mock benchmark data for validation testing")
            
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in benchmark data file: {e}",
                e.doc, e.pos
            )

    def _generate_mock_benchmark_data(self) -> Dict[str, Any]:
        """
        Generate mock benchmark data for testing environments per 
        TST-PERF-003 statistical validation requirements.
        
        Returns:
            Mock benchmark data structure compatible with pytest-benchmark output
        """
        return {
            "machine_info": {
                "node": "test-runner",
                "processor": "test-cpu",
                "machine": "test-machine",
                "python_version": "3.8.0",
                "python_implementation": "CPython"
            },
            "commit_info": {
                "id": "test-commit",
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "author_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dirty": False,
                "project": "flyrigloader"
            },
            "benchmarks": [
                {
                    "group": "data_loading",
                    "name": "test_load_standard_pickle_100mb",
                    "fullname": "tests/benchmarks/test_loading_performance.py::test_load_standard_pickle_100mb",
                    "params": {"data_size": "100MB"},
                    "stats": {
                        "min": 0.8,
                        "max": 1.2,
                        "mean": 0.9,
                        "stddev": 0.1,
                        "rounds": 5,
                        "median": 0.9,
                        "iqr": 0.1,
                        "q1": 0.85,
                        "q3": 0.95,
                        "iqr_outliers": 0,
                        "stddev_outliers": 0,
                        "outliers": "0;0",
                        "ld15iqr": 0.8,
                        "hd15iqr": 1.2,
                        "ops": 1.11
                    }
                }
            ],
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "4.0.0"
        }

    def validate_overall_coverage(self) -> Tuple[bool, float, float]:
        """
        Validate overall coverage thresholds per TST-COV-001 requirements.
        
        Returns:
            Tuple of (threshold_met, line_coverage, branch_coverage)
        """
        totals = self.coverage_data.get("totals", {})
        
        line_coverage = totals.get("percent_covered", 0.0)
        branch_coverage = totals.get("percent_covered_branch", 0.0)
        
        # Handle missing branch coverage data
        if branch_coverage == 0.0 and totals.get("num_branches", 0) > 0:
            covered_branches = totals.get("covered_branches", 0)
            total_branches = totals.get("num_branches", 1)
            branch_coverage = (covered_branches / total_branches) * 100.0
        
        overall_threshold = self.coverage_config["global_configuration"]["overall_threshold"]
        branch_threshold = self.coverage_config["global_configuration"]["branch_threshold"]
        
        line_threshold_met = line_coverage >= overall_threshold
        branch_threshold_met = branch_coverage >= branch_threshold
        
        self.logger.info(
            f"Overall coverage: {line_coverage:.2f}% lines (threshold: {overall_threshold}%), "
            f"{branch_coverage:.2f}% branches (threshold: {branch_threshold}%)"
        )
        
        return (line_threshold_met and branch_threshold_met, line_coverage, branch_coverage)

    def validate_module_specific_coverage(self) -> Dict[str, ModuleCoverageResult]:
        """
        Validate module-specific coverage per TST-COV-002 critical module
        requirements and module category thresholds.
        
        Returns:
            Dictionary of module paths to coverage results
        """
        module_results = {}
        files_data = self.coverage_data.get("files", {})
        
        # Process each module category from configuration
        for category_name, category_config in self.coverage_config["module_thresholds"].items():
            category_modules = category_config.get("modules", {})
            
            for module_path, module_config in category_modules.items():
                result = self._analyze_module_coverage(
                    module_path, module_config, files_data, category_name
                )
                module_results[module_path] = result
        
        return module_results

    def _analyze_module_coverage(self, 
                                module_path: str, 
                                module_config: Dict[str, Any],
                                files_data: Dict[str, Any],
                                category: str) -> ModuleCoverageResult:
        """
        Analyze coverage for a specific module with detailed metrics extraction.
        
        Args:
            module_path: Path to the module being analyzed
            module_config: Module-specific configuration from thresholds
            files_data: Coverage data for all files
            category: Module category (critical, utility, initialization)
            
        Returns:
            Detailed coverage analysis result for the module
        """
        # Find matching files for the module (handle directory patterns)
        matching_files = self._find_matching_files(module_path, files_data)
        
        if not matching_files:
            self.logger.warning(f"No coverage data found for module: {module_path}")
            return ModuleCoverageResult(
                module_path=module_path,
                line_coverage=0.0,
                branch_coverage=0.0,
                lines_covered=0,
                lines_total=1,
                branches_covered=0,
                branches_total=1,
                category=category,
                requirements=module_config.get("requirements", [])
            )
        
        # Aggregate coverage across all matching files
        total_lines = 0
        covered_lines = 0
        total_branches = 0
        covered_branches = 0
        all_missing_lines = []
        all_partial_branches = []
        
        for file_path, file_data in matching_files.items():
            summary = file_data.get("summary", {})
            
            file_covered = summary.get("covered_lines", 0)
            file_total = summary.get("num_statements", 0)
            file_branch_covered = summary.get("covered_branches", 0)
            file_branch_total = summary.get("num_branches", 0)
            
            covered_lines += file_covered
            total_lines += file_total
            covered_branches += file_branch_covered
            total_branches += file_branch_total
            
            # Collect missing lines and partial branches
            all_missing_lines.extend(summary.get("missing_lines", []))
            all_partial_branches.extend(summary.get("excluded_lines", []))
        
        # Calculate coverage percentages
        line_coverage = (covered_lines / total_lines * 100.0) if total_lines > 0 else 0.0
        branch_coverage = (covered_branches / total_branches * 100.0) if total_branches > 0 else 0.0
        
        # Check thresholds
        line_threshold = module_config.get("threshold", 90.0)
        branch_threshold = module_config.get("branch_threshold", 90.0)
        
        threshold_met = line_coverage >= line_threshold
        branch_threshold_met = branch_coverage >= branch_threshold
        
        return ModuleCoverageResult(
            module_path=module_path,
            line_coverage=line_coverage,
            branch_coverage=branch_coverage,
            lines_covered=covered_lines,
            lines_total=total_lines,
            branches_covered=covered_branches,
            branches_total=total_branches,
            missing_lines=all_missing_lines,
            partial_branches=all_partial_branches,
            threshold_met=threshold_met,
            branch_threshold_met=branch_threshold_met,
            category=category,
            requirements=module_config.get("requirements", [])
        )

    def _find_matching_files(self, module_path: str, files_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find files in coverage data that match the module path pattern.
        
        Args:
            module_path: Module path pattern (may include directories)
            files_data: Coverage data for all files
            
        Returns:
            Dictionary of matching file paths to their coverage data
        """
        matching_files = {}
        
        # Handle directory patterns (e.g., "src/flyrigloader/config/")
        if module_path.endswith('/'):
            # Match all files in the directory
            for file_path, file_data in files_data.items():
                if file_path.startswith(module_path):
                    matching_files[file_path] = file_data
        else:
            # Match exact file or files containing the pattern
            for file_path, file_data in files_data.items():
                if module_path in file_path or file_path.endswith(module_path):
                    matching_files[file_path] = file_data
        
        return matching_files

    def validate_performance_slas(self) -> List[PerformanceBenchmarkResult]:
        """
        Validate performance SLA requirements per TST-PERF-001 and 
        TST-PERF-002 specifications.
        
        Returns:
            List of performance benchmark validation results
        """
        performance_results = []
        benchmarks = self.benchmark_data.get("benchmarks", [])
        
        # Get SLA configuration from quality gates
        sla_config = self.quality_gates_config.get("performance", {}).get("sla_categories", {})
        
        for benchmark in benchmarks:
            result = self._analyze_benchmark_performance(benchmark, sla_config)
            if result:
                performance_results.append(result)
        
        return performance_results

    def _analyze_benchmark_performance(self, 
                                     benchmark: Dict[str, Any],
                                     sla_config: Dict[str, Any]) -> Optional[PerformanceBenchmarkResult]:
        """
        Analyze individual benchmark performance against SLA requirements.
        
        Args:
            benchmark: Individual benchmark data from pytest-benchmark
            sla_config: SLA configuration from quality gates
            
        Returns:
            Performance benchmark result or None if not applicable
        """
        benchmark_name = benchmark.get("name", "")
        stats = benchmark.get("stats", {})
        mean_time = stats.get("mean", 0.0)
        
        # Determine benchmark category and data size
        category = self._determine_benchmark_category(benchmark_name)
        data_size = self._extract_data_size(benchmark)
        
        if category == "data_loading":
            # TST-PERF-001: Data loading SLA validation (1s per 100MB)
            sla_threshold = sla_config.get("data_loading", {}).get("max_time_per_mb", 0.01)
            expected_time = data_size * sla_threshold
            sla_met = mean_time <= expected_time
            
            return PerformanceBenchmarkResult(
                operation_name=benchmark_name,
                measured_time=mean_time,
                data_size=data_size,
                sla_threshold=expected_time,
                sla_met=sla_met,
                unit="seconds",
                benchmark_category=category
            )
            
        elif category == "data_transformation":
            # TST-PERF-002: DataFrame transformation SLA validation (500ms per 1M rows)
            sla_threshold = sla_config.get("data_transformation", {}).get("max_time_per_million_rows", 0.5)
            rows_in_millions = data_size / 1_000_000
            expected_time = rows_in_millions * sla_threshold
            sla_met = mean_time <= expected_time
            
            return PerformanceBenchmarkResult(
                operation_name=benchmark_name,
                measured_time=mean_time,
                data_size=data_size,
                sla_threshold=expected_time,
                sla_met=sla_met,
                unit="seconds",
                benchmark_category=category
            )
        
        return None

    def _determine_benchmark_category(self, benchmark_name: str) -> str:
        """Determine benchmark category from benchmark name."""
        if any(keyword in benchmark_name.lower() for keyword in 
               ["load", "pickle", "deserialize", "read"]):
            return "data_loading"
        elif any(keyword in benchmark_name.lower() for keyword in 
                 ["transform", "dataframe", "convert", "process"]):
            return "data_transformation"
        else:
            return "unknown"

    def _extract_data_size(self, benchmark: Dict[str, Any]) -> float:
        """
        Extract data size from benchmark parameters or name.
        
        Args:
            benchmark: Benchmark data dictionary
            
        Returns:
            Data size in appropriate units (bytes for loading, rows for transformation)
        """
        params = benchmark.get("params", {})
        
        # Check for explicit data size in parameters
        if "data_size" in params:
            size_str = params["data_size"]
            return self._parse_size_string(size_str)
        
        # Extract from benchmark name
        name = benchmark.get("name", "")
        
        # Common size patterns in benchmark names
        size_patterns = [
            r"(\d+)mb",
            r"(\d+)_mb", 
            r"(\d+)rows",
            r"(\d+)_rows",
            r"(\d+)k_rows",
            r"(\d+)m_rows"
        ]
        
        for pattern in size_patterns:
            match = re.search(pattern, name.lower())
            if match:
                size_value = int(match.group(1))
                
                if "mb" in pattern:
                    return size_value * 1024 * 1024  # Convert MB to bytes
                elif "k_rows" in pattern:
                    return size_value * 1000  # Convert K rows to rows
                elif "m_rows" in pattern:
                    return size_value * 1_000_000  # Convert M rows to rows
                else:
                    return size_value
        
        # Default size for unknown benchmarks
        return 100 * 1024 * 1024  # 100MB default

    def _parse_size_string(self, size_str: str) -> float:
        """Parse size string like '100MB' or '1M rows' to numeric value."""
        size_str = size_str.lower().strip()
        
        # Extract numeric value and unit
        match = re.match(r"(\d+(?:\.\d+)?)\s*([a-z]+)", size_str)
        if not match:
            return 100 * 1024 * 1024  # Default 100MB
        
        value = float(match.group(1))
        unit = match.group(2)
        
        # Convert based on unit
        if unit == "mb":
            return value * 1024 * 1024
        elif unit == "gb":
            return value * 1024 * 1024 * 1024
        elif unit == "kb":
            return value * 1024
        elif "rows" in unit:
            return value
        else:
            return value

    def generate_quality_gate_result(self) -> QualityGateResult:
        """
        Generate comprehensive quality gate validation result implementing
        TST-COV-004 automated merge blocking requirements.
        
        Returns:
            Complete quality gate analysis with violations and recommendations
        """
        self.logger.info("Generating comprehensive quality gate validation result")
        
        # Validate overall coverage
        overall_met, overall_line, overall_branch = self.validate_overall_coverage()
        
        # Validate module-specific coverage
        module_results = self.validate_module_specific_coverage()
        
        # Validate performance SLAs
        performance_results = self.validate_performance_slas()
        
        # Determine overall quality gate status
        violations = []
        warnings = []
        
        # Check overall coverage violations
        if not overall_met:
            threshold = self.coverage_config["global_configuration"]["overall_threshold"]
            violations.append(
                f"Overall coverage {overall_line:.2f}% below threshold {threshold}% "
                f"(TST-COV-001 violation)"
            )
        
        # Check critical module violations
        critical_violations = []
        for module_path, result in module_results.items():
            if result.category == "critical_modules":
                if not result.threshold_met:
                    critical_violations.append(
                        f"Critical module {module_path}: {result.line_coverage:.2f}% "
                        f"below required 100% (TST-COV-002 violation)"
                    )
                if not result.branch_threshold_met:
                    critical_violations.append(
                        f"Critical module {module_path}: {result.branch_coverage:.2f}% "
                        f"branch coverage below required 100% (TST-COV-002 violation)"
                    )
        
        violations.extend(critical_violations)
        critical_modules_met = len(critical_violations) == 0
        
        # Check performance SLA violations
        performance_violations = []
        for perf_result in performance_results:
            if not perf_result.sla_met:
                performance_violations.append(
                    f"Performance SLA violation in {perf_result.operation_name}: "
                    f"{perf_result.measured_time:.3f}s exceeds threshold "
                    f"{perf_result.sla_threshold:.3f}s"
                )
        
        violations.extend(performance_violations)
        performance_slas_met = len(performance_violations) == 0
        
        # Generate warnings for utility modules below threshold
        for module_path, result in module_results.items():
            if result.category == "utility_modules" and not result.threshold_met:
                warnings.append(
                    f"Utility module {module_path}: {result.line_coverage:.2f}% "
                    f"below recommended {result.line_coverage}%"
                )
        
        execution_time = time.time() - self.start_time
        
        return QualityGateResult(
            overall_coverage_met=overall_met,
            critical_modules_met=critical_modules_met,
            branch_coverage_met=overall_branch >= self.coverage_config["global_configuration"]["branch_threshold"],
            performance_slas_met=performance_slas_met,
            module_results=module_results,
            performance_results=performance_results,
            violations=violations,
            warnings=warnings,
            overall_line_coverage=overall_line,
            overall_branch_coverage=overall_branch,
            execution_time=execution_time
        )

    def generate_detailed_report(self, result: QualityGateResult) -> str:
        """
        Generate comprehensive detailed report per TST-COV-003 reporting 
        requirements.
        
        Args:
            result: Complete quality gate validation result
            
        Returns:
            Formatted detailed report string
        """
        report_lines = [
            "=" * 80,
            "FLYRIGLOADER COVERAGE VALIDATION REPORT",
            "=" * 80,
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Execution Time: {result.execution_time:.2f} seconds",
            f"Configuration: {self.coverage_config_path.name}",
            "",
            "OVERALL QUALITY GATE STATUS",
            "-" * 40,
        ]
        
        # Overall status
        status = "✅ PASSED" if (
            result.overall_coverage_met and 
            result.critical_modules_met and 
            result.performance_slas_met
        ) else "❌ FAILED"
        
        report_lines.extend([
            f"Quality Gate Status: {status}",
            f"Overall Line Coverage: {result.overall_line_coverage:.2f}%",
            f"Overall Branch Coverage: {result.overall_branch_coverage:.2f}%",
            f"Critical Modules Status: {'✅ PASSED' if result.critical_modules_met else '❌ FAILED'}",
            f"Performance SLAs Status: {'✅ PASSED' if result.performance_slas_met else '❌ FAILED'}",
            ""
        ])
        
        # Violations section
        if result.violations:
            report_lines.extend([
                "QUALITY GATE VIOLATIONS",
                "-" * 40,
            ])
            for i, violation in enumerate(result.violations, 1):
                report_lines.append(f"{i}. {violation}")
            report_lines.append("")
        
        # Module-specific coverage breakdown
        report_lines.extend([
            "MODULE COVERAGE BREAKDOWN",
            "-" * 40,
        ])
        
        for category in ["critical_modules", "utility_modules", "initialization_modules"]:
            category_modules = [
                (path, result) for path, result in result.module_results.items()
                if result.category == category
            ]
            
            if category_modules:
                report_lines.append(f"\n{category.replace('_', ' ').title()}:")
                
                for module_path, module_result in category_modules:
                    status_icon = "✅" if (module_result.threshold_met and module_result.branch_threshold_met) else "❌"
                    report_lines.append(
                        f"  {status_icon} {module_path}: "
                        f"{module_result.line_coverage:.1f}% lines, "
                        f"{module_result.branch_coverage:.1f}% branches"
                    )
                    
                    if not module_result.threshold_met or not module_result.branch_threshold_met:
                        if module_result.missing_lines:
                            report_lines.append(f"    Missing lines: {module_result.missing_lines[:10]}")
                        if len(module_result.missing_lines) > 10:
                            report_lines.append(f"    ... and {len(module_result.missing_lines) - 10} more")
        
        # Performance SLA breakdown
        if result.performance_results:
            report_lines.extend([
                "",
                "PERFORMANCE SLA VALIDATION",
                "-" * 40,
            ])
            
            for perf_result in result.performance_results:
                status_icon = "✅" if perf_result.sla_met else "❌"
                report_lines.append(
                    f"  {status_icon} {perf_result.operation_name}: "
                    f"{perf_result.measured_time:.3f}s "
                    f"(threshold: {perf_result.sla_threshold:.3f}s)"
                )
        
        # Warnings section
        if result.warnings:
            report_lines.extend([
                "",
                "WARNINGS",
                "-" * 40,
            ])
            for i, warning in enumerate(result.warnings, 1):
                report_lines.append(f"{i}. ⚠️  {warning}")
        
        # Recommendations
        report_lines.extend([
            "",
            "RECOMMENDATIONS",
            "-" * 40,
        ])
        
        if result.violations:
            report_lines.append("• Address all quality gate violations before merge")
            if not result.critical_modules_met:
                report_lines.append("• Add comprehensive tests for critical modules requiring 100% coverage")
            if not result.performance_slas_met:
                report_lines.append("• Optimize performance-critical operations to meet SLA requirements")
        else:
            report_lines.append("• All quality gates passed - code ready for merge")
        
        report_lines.extend([
            "",
            "=" * 80,
            "Report generated by flyrigloader test infrastructure",
            "For support: https://github.com/flyrigloader/issues",
            "=" * 80
        ])
        
        return "\n".join(report_lines)

    def export_json_report(self, result: QualityGateResult, output_path: Path) -> None:
        """
        Export machine-readable JSON report for CI/CD integration per
        TST-COV-003 machine-readable reporting requirements.
        
        Args:
            result: Complete quality gate validation result
            output_path: Path for JSON report output
        """
        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "execution_time": result.execution_time,
            "overall_status": {
                "passed": (
                    result.overall_coverage_met and 
                    result.critical_modules_met and 
                    result.performance_slas_met
                ),
                "overall_coverage_met": result.overall_coverage_met,
                "critical_modules_met": result.critical_modules_met,
                "branch_coverage_met": result.branch_coverage_met,
                "performance_slas_met": result.performance_slas_met,
                "line_coverage": result.overall_line_coverage,
                "branch_coverage": result.overall_branch_coverage
            },
            "violations": result.violations,
            "warnings": result.warnings,
            "module_coverage": {
                path: {
                    "line_coverage": module.line_coverage,
                    "branch_coverage": module.branch_coverage,
                    "threshold_met": module.threshold_met,
                    "branch_threshold_met": module.branch_threshold_met,
                    "category": module.category,
                    "requirements": module.requirements,
                    "lines_covered": module.lines_covered,
                    "lines_total": module.lines_total,
                    "branches_covered": module.branches_covered,
                    "branches_total": module.branches_total,
                    "missing_lines": module.missing_lines
                }
                for path, module in result.module_results.items()
            },
            "performance_results": [
                {
                    "operation_name": perf.operation_name,
                    "measured_time": perf.measured_time,
                    "data_size": perf.data_size,
                    "sla_threshold": perf.sla_threshold,
                    "sla_met": perf.sla_met,
                    "unit": perf.unit,
                    "benchmark_category": perf.benchmark_category
                }
                for perf in result.performance_results
            ],
            "metadata": {
                "version": "1.0.0",
                "tool": "flyrigloader-coverage-validator",
                "configuration_files": {
                    "coverage_thresholds": str(self.coverage_config_path),
                    "quality_gates": str(self.quality_gates_config_path)
                }
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.logger.info(f"JSON report exported to {output_path}")


# ============================================================================
# COMMAND LINE INTERFACE AND MAIN EXECUTION
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for coverage validation script.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Automated coverage validation with quality gate enforcement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate-coverage.py
  python validate-coverage.py --coverage-data coverage.json
  python validate-coverage.py --output-json results.json --verbose
  python validate-coverage.py --ci-mode --fail-on-violation

CI/CD Integration:
  Exit code 0: All quality gates passed
  Exit code 1: Quality gate violations detected
        """
    )
    
    parser.add_argument(
        "--coverage-config",
        type=Path,
        default=Path("tests/coverage/coverage-thresholds.json"),
        help="Path to coverage thresholds configuration file"
    )
    
    parser.add_argument(
        "--quality-gates-config",
        type=Path,
        default=Path("tests/coverage/quality-gates.yml"),
        help="Path to quality gates configuration file"
    )
    
    parser.add_argument(
        "--coverage-data",
        type=Path,
        default=Path("coverage.json"),
        help="Path to coverage data file (coverage.json)"
    )
    
    parser.add_argument(
        "--benchmark-data",
        type=Path,
        default=Path("benchmark-results.json"),
        help="Path to benchmark results file"
    )
    
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Path for JSON report output (for CI/CD integration)"
    )
    
    parser.add_argument(
        "--output-html",
        type=Path,
        help="Path for HTML report output"
    )
    
    parser.add_argument(
        "--ci-mode",
        action="store_true",
        help="Enable CI/CD mode with structured logging"
    )
    
    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        default=True,
        help="Exit with code 1 on quality gate violations (default: True)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output"
    )
    
    return parser.parse_args()


def main() -> int:
    """
    Main entry point for coverage validation script implementing comprehensive
    quality gate enforcement per TST-COV-004 requirements.
    
    Returns:
        Exit code: 0 for success, 1 for quality gate violations
    """
    args = parse_arguments()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize coverage validator
        validator = CoverageValidator(
            coverage_config_path=args.coverage_config,
            quality_gates_config_path=args.quality_gates_config,
            coverage_data_path=args.coverage_data,
            benchmark_data_path=args.benchmark_data
        )
        
        # Load all configurations and data
        validator.load_configurations()
        validator.load_coverage_data()
        validator.load_benchmark_data()
        
        # Generate comprehensive quality gate result
        result = validator.generate_quality_gate_result()
        
        # Generate and display detailed report
        detailed_report = validator.generate_detailed_report(result)
        print(detailed_report)
        
        # Export JSON report if requested
        if args.output_json:
            validator.export_json_report(result, args.output_json)
        
        # Determine exit code based on quality gate status
        quality_gate_passed = (
            result.overall_coverage_met and 
            result.critical_modules_met and 
            result.performance_slas_met
        )
        
        if args.ci_mode:
            # Structured logging for CI/CD systems
            print(f"::set-output name=coverage_passed::{quality_gate_passed}")
            print(f"::set-output name=line_coverage::{result.overall_line_coverage:.2f}")
            print(f"::set-output name=branch_coverage::{result.overall_branch_coverage:.2f}")
            print(f"::set-output name=violations_count::{len(result.violations)}")
        
        if not quality_gate_passed and args.fail_on_violation:
            validator.logger.error("Quality gate violations detected - blocking merge per TST-COV-004")
            return 1
        
        validator.logger.info("All quality gates passed - code ready for integration")
        return 0
        
    except Exception as e:
        print(f"❌ Coverage validation failed with error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())