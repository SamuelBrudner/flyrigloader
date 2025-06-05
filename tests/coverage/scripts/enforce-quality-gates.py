#!/usr/bin/env python3
"""
Comprehensive Quality Gate Enforcement Script

Implements automated validation of coverage thresholds, performance SLAs, and test 
execution criteria with CI/CD integration and merge blocking capabilities. Serves 
as the primary quality assurance gatekeeper per TST-COV-004 and Section 4.1.1.5 
requirements.

This script orchestrates:
- TST-COV-001: Maintain >90% overall test coverage across all modules
- TST-COV-002: Achieve 100% coverage for critical data loading and validation modules  
- TST-COV-004: Block merges when coverage drops below thresholds
- TST-PERF-001: Data loading SLA validation within 1s per 100MB
- TST-PERF-002: DataFrame transformation SLA validation within 500ms per 1M rows
- Section 4.1.1.5: Test execution workflow with automated quality gate enforcement

Integration Points:
- validate-coverage.py for comprehensive coverage analysis
- check-performance-slas.py for performance validation  
- quality-gates.yml configuration for enforcement thresholds
- CI/CD pipeline integration with detailed status reporting

Authors: FlyRigLoader Test Infrastructure Team
Created: 2024-12-19
Last Modified: 2024-12-19
Version: 1.0.0
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


# ============================================================================
# CONFIGURATION AND DATA STRUCTURES
# ============================================================================

@dataclass
class QualityGateCategory:
    """
    Represents a quality gate category with validation rules and thresholds.
    
    Implements comprehensive validation criteria per Section 4.1.1.5 test 
    execution workflow requirements.
    """
    name: str
    weight: float  # Percentage weight in overall quality score
    critical: bool  # Whether failure blocks deployment
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    passed: bool = False
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityGateValidationResult:
    """
    Comprehensive quality gate validation result with detailed analysis.
    
    Aggregates all category results with actionable feedback per TST-COV-004 
    automated merge blocking requirements.
    """
    overall_passed: bool
    quality_score: float  # Weighted score across all categories
    categories: Dict[str, QualityGateCategory] = field(default_factory=dict)
    total_violations: int = 0
    total_warnings: int = 0
    critical_failures: List[str] = field(default_factory=list)
    execution_summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    merge_allowed: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# QUALITY GATE ORCHESTRATION ENGINE
# ============================================================================

class QualityGateEnforcer:
    """
    Primary quality gate enforcement engine implementing comprehensive validation 
    orchestration with automated merge blocking per TST-COV-004 requirements.
    
    Coordinates coverage validation, performance SLA checking, and test execution
    criteria with detailed reporting and CI/CD pipeline integration per Section 
    4.1.1.5 test execution workflow.
    """

    def __init__(self, 
                 quality_gates_config_path: Path = Path("tests/coverage/quality-gates.yml"),
                 coverage_config_path: Path = Path("tests/coverage/coverage-thresholds.json"),
                 workspace_root: Optional[Path] = None):
        """
        Initialize quality gate enforcer with configuration and workspace paths.
        
        Args:
            quality_gates_config_path: Path to quality gates configuration
            coverage_config_path: Path to coverage thresholds configuration
            workspace_root: Root workspace directory (defaults to current directory)
        """
        self.quality_gates_config_path = quality_gates_config_path
        self.coverage_config_path = coverage_config_path
        self.workspace_root = workspace_root or Path.cwd()
        
        # Configuration storage
        self.quality_gates_config: Dict[str, Any] = {}
        self.coverage_config: Dict[str, Any] = {}
        
        # Validation script paths
        self.validate_coverage_script = self.workspace_root / "tests/coverage/validate-coverage.py"
        self.check_performance_script = self.workspace_root / "tests/coverage/scripts/check-performance-slas.py"
        
        # Results storage
        self.validation_result: Optional[QualityGateValidationResult] = None
        
        # Logging configuration with comprehensive quality gate tracking
        self.logger = self._configure_logging()
        
        # Performance tracking
        self.start_time = time.time()
        
        self.logger.info("Initialized QualityGateEnforcer for comprehensive validation orchestration")

    def _configure_logging(self) -> logging.Logger:
        """
        Configure comprehensive logging for quality gate enforcement tracking.
        
        Returns:
            Configured logger with detailed formatting for CI/CD integration
        """
        logger = logging.getLogger("quality_gate_enforcer")
        logger.setLevel(logging.INFO)
        
        # Create console handler with CI/CD-friendly formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Enhanced formatter for quality gate status tracking
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        
        # File handler for detailed audit logging
        log_dir = self.workspace_root / "tests/coverage/logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / "quality-gate-enforcement.log", 
            mode="a"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Avoid duplicate handlers in testing environments
        if not logger.handlers:
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        
        return logger

    def load_configurations(self) -> None:
        """
        Load and validate quality gates and coverage configuration files.
        
        Implements comprehensive configuration validation per Section 4.1.1.5
        quality assurance requirements.
        
        Raises:
            FileNotFoundError: If configuration files are missing
            yaml.YAMLError: If YAML configuration is invalid
            json.JSONDecodeError: If JSON configuration is invalid
        """
        self.logger.info("Loading quality gate enforcement configurations")
        
        # Load quality gates configuration
        try:
            with open(self.quality_gates_config_path, 'r') as f:
                self.quality_gates_config = yaml.safe_load(f)
            
            self.logger.info(f"Loaded quality gates configuration from {self.quality_gates_config_path}")
            
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Quality gates configuration not found: {self.quality_gates_config_path}"
            )
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in quality gates configuration: {e}")
        
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
        
        # Validate configuration integrity
        self._validate_configuration_integrity()

    def _validate_configuration_integrity(self) -> None:
        """
        Validate configuration file integrity and consistency per Section 4.1.1.5 
        quality assurance requirements.
        
        Raises:
            ValueError: If configuration validation fails
        """
        # Validate required quality gates configuration keys
        required_quality_keys = ["coverage", "performance", "execution", "categories"]
        for key in required_quality_keys:
            if key not in self.quality_gates_config:
                raise ValueError(f"Missing required quality gates configuration key: {key}")
        
        # Validate required coverage configuration keys  
        required_coverage_keys = ["global_settings", "critical_modules", "quality_gates"]
        for key in required_coverage_keys:
            if key not in self.coverage_config:
                raise ValueError(f"Missing required coverage configuration key: {key}")
        
        # Validate threshold consistency between configurations
        quality_threshold = self.quality_gates_config["coverage"]["overall_coverage_threshold"]
        coverage_threshold = self.coverage_config["global_settings"]["overall_threshold"]["line_coverage"]
        
        if abs(quality_threshold - coverage_threshold) > 0.1:
            self.logger.warning(
                f"Coverage threshold mismatch: quality-gates.yml={quality_threshold}%, "
                f"coverage-thresholds.json={coverage_threshold}%"
            )
        
        # Validate script dependencies exist
        required_scripts = [self.validate_coverage_script, self.check_performance_script]
        for script_path in required_scripts:
            if not script_path.exists():
                raise FileNotFoundError(f"Required validation script not found: {script_path}")
        
        self.logger.info("Configuration integrity validation completed successfully")

    def validate_coverage_quality_gate(self) -> QualityGateCategory:
        """
        Execute coverage validation quality gate implementing TST-COV-001, 
        TST-COV-002, and TST-COV-004 requirements.
        
        Returns:
            QualityGateCategory with comprehensive coverage validation results
        """
        self.logger.info("Executing coverage quality gate validation")
        category_start_time = time.time()
        
        # Initialize coverage quality gate category
        coverage_category = QualityGateCategory(
            name="coverage",
            weight=40.0,  # 40% weight in overall quality score
            critical=True,  # Coverage failures block deployment
            validation_rules=self.quality_gates_config["categories"]["coverage_validation"]["validation_rules"]
        )
        
        try:
            # Execute validate-coverage.py script
            coverage_cmd = [
                sys.executable,
                str(self.validate_coverage_script),
                "--coverage-config", str(self.coverage_config_path),
                "--quality-gates-config", str(self.quality_gates_config_path),
                "--output-json", str(self.workspace_root / "coverage-validation-results.json"),
                "--ci-mode",
                "--fail-on-violation"
            ]
            
            self.logger.info(f"Executing coverage validation: {' '.join(coverage_cmd)}")
            
            result = subprocess.run(
                coverage_cmd,
                capture_output=True,
                text=True,
                cwd=self.workspace_root,
                timeout=300  # 5 minute timeout
            )
            
            # Parse coverage validation results
            coverage_results = self._parse_coverage_results(result)
            
            # Apply validation rules
            self._apply_coverage_validation_rules(coverage_category, coverage_results)
            
            # Determine overall coverage category status
            coverage_category.passed = (
                result.returncode == 0 and 
                len(coverage_category.violations) == 0
            )
            
            self.logger.info(
                f"Coverage quality gate: {'PASSED' if coverage_category.passed else 'FAILED'} "
                f"({len(coverage_category.violations)} violations)"
            )
            
        except subprocess.TimeoutExpired:
            coverage_category.violations.append(
                "Coverage validation timed out after 5 minutes (TST-INF-002 violation)"
            )
            self.logger.error("Coverage validation exceeded timeout threshold")
            
        except Exception as e:
            coverage_category.violations.append(f"Coverage validation execution failed: {e}")
            self.logger.error(f"Coverage validation execution error: {e}")
        
        # Record execution time
        coverage_category.execution_time = time.time() - category_start_time
        
        return coverage_category

    def _parse_coverage_results(self, result: subprocess.CompletedProcess) -> Dict[str, Any]:
        """
        Parse coverage validation results from subprocess execution.
        
        Args:
            result: Completed process result from coverage validation
            
        Returns:
            Parsed coverage validation results dictionary
        """
        coverage_results = {
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "overall_coverage": 0.0,
            "critical_modules_passed": False,
            "violations": []
        }
        
        # Attempt to load JSON results if available
        json_results_path = self.workspace_root / "coverage-validation-results.json"
        if json_results_path.exists():
            try:
                with open(json_results_path, 'r') as f:
                    json_results = json.load(f)
                
                coverage_results.update({
                    "overall_coverage": json_results.get("overall_status", {}).get("line_coverage", 0.0),
                    "critical_modules_passed": json_results.get("overall_status", {}).get("critical_modules_met", False),
                    "violations": json_results.get("violations", [])
                })
                
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.warning(f"Failed to parse coverage JSON results: {e}")
        
        # Parse coverage metrics from stdout if JSON not available
        if coverage_results["overall_coverage"] == 0.0 and result.stdout:
            coverage_results["overall_coverage"] = self._extract_coverage_from_output(result.stdout)
        
        return coverage_results

    def _extract_coverage_from_output(self, output: str) -> float:
        """
        Extract coverage percentage from command output text.
        
        Args:
            output: Command output text
            
        Returns:
            Extracted coverage percentage (0.0 if not found)
        """
        import re
        
        # Look for coverage percentage patterns in output
        coverage_patterns = [
            r"Overall.*?(\d+(?:\.\d+)?)%",
            r"coverage.*?(\d+(?:\.\d+)?)%",
            r"(\d+(?:\.\d+)?)%.*coverage"
        ]
        
        for pattern in coverage_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        return 0.0

    def _apply_coverage_validation_rules(self, 
                                       category: QualityGateCategory, 
                                       results: Dict[str, Any]) -> None:
        """
        Apply coverage validation rules per TST-COV-001, TST-COV-002, and TST-COV-004.
        
        Args:
            category: Coverage quality gate category to update
            results: Coverage validation results
        """
        overall_coverage = results["overall_coverage"]
        critical_modules_passed = results["critical_modules_passed"]
        
        # TST-COV-001: Overall coverage threshold validation
        overall_threshold = self.quality_gates_config["coverage"]["overall_coverage_threshold"]
        if overall_coverage < overall_threshold:
            violation = (
                f"Overall coverage {overall_coverage:.2f}% below required {overall_threshold}% "
                f"(TST-COV-001 violation)"
            )
            category.violations.append(violation)
        
        # TST-COV-002: Critical module coverage validation
        if not critical_modules_passed:
            violation = (
                "Critical module coverage below required 100% threshold "
                "(TST-COV-002 violation)"
            )
            category.violations.append(violation)
        
        # Add violations from detailed validation
        if results["violations"]:
            category.violations.extend(results["violations"])
        
        # Store metadata for reporting
        category.metadata.update({
            "overall_coverage": overall_coverage,
            "critical_modules_passed": critical_modules_passed,
            "coverage_threshold": overall_threshold,
            "return_code": results["return_code"]
        })

    def validate_performance_quality_gate(self) -> QualityGateCategory:
        """
        Execute performance SLA validation quality gate implementing TST-PERF-001 
        and TST-PERF-002 requirements.
        
        Returns:
            QualityGateCategory with comprehensive performance validation results
        """
        self.logger.info("Executing performance quality gate validation")
        category_start_time = time.time()
        
        # Initialize performance quality gate category
        performance_category = QualityGateCategory(
            name="performance",
            weight=35.0,  # 35% weight in overall quality score
            critical=True,  # Performance failures block deployment
            validation_rules=self.quality_gates_config["categories"]["performance_validation"]["validation_rules"]
        )
        
        try:
            # Execute check-performance-slas.py script
            performance_cmd = [
                sys.executable,
                str(self.check_performance_script),
                "--quality-gates", str(self.quality_gates_config_path),
                "--output-format", "json",
                "--fail-on-violation",
                "--verbose"
            ]
            
            self.logger.info(f"Executing performance validation: {' '.join(performance_cmd)}")
            
            result = subprocess.run(
                performance_cmd,
                capture_output=True,
                text=True,
                cwd=self.workspace_root,
                timeout=600  # 10 minute timeout for performance benchmarks
            )
            
            # Parse performance validation results
            performance_results = self._parse_performance_results(result)
            
            # Apply validation rules
            self._apply_performance_validation_rules(performance_category, performance_results)
            
            # Determine overall performance category status
            performance_category.passed = (
                result.returncode == 0 and 
                len(performance_category.violations) == 0
            )
            
            self.logger.info(
                f"Performance quality gate: {'PASSED' if performance_category.passed else 'FAILED'} "
                f"({len(performance_category.violations)} violations)"
            )
            
        except subprocess.TimeoutExpired:
            performance_category.violations.append(
                "Performance validation timed out after 10 minutes (TST-INF-002 violation)"
            )
            self.logger.error("Performance validation exceeded timeout threshold")
            
        except Exception as e:
            performance_category.violations.append(f"Performance validation execution failed: {e}")
            self.logger.error(f"Performance validation execution error: {e}")
        
        # Record execution time
        performance_category.execution_time = time.time() - category_start_time
        
        return performance_category

    def _parse_performance_results(self, result: subprocess.CompletedProcess) -> Dict[str, Any]:
        """
        Parse performance validation results from subprocess execution.
        
        Args:
            result: Completed process result from performance validation
            
        Returns:
            Parsed performance validation results dictionary
        """
        performance_results = {
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "sla_violations": 0,
            "performance_regressions": 0,
            "data_loading_sla_met": True,
            "transformation_sla_met": True,
            "violations": []
        }
        
        # Look for generated performance report JSON files
        reports_dir = self.workspace_root / "tests/coverage/reports"
        if reports_dir.exists():
            json_files = list(reports_dir.glob("performance_sla_report_*.json"))
            if json_files:
                # Use the most recent report
                latest_report = max(json_files, key=lambda f: f.stat().st_mtime)
                
                try:
                    with open(latest_report, 'r') as f:
                        json_results = json.load(f)
                    
                    performance_results.update({
                        "sla_violations": json_results.get("total_violations", 0),
                        "performance_regressions": json_results.get("regression_count", 0),
                        "overall_status": json_results.get("overall_status", "UNKNOWN"),
                        "quality_gate_status": json_results.get("quality_gate_status", "UNKNOWN")
                    })
                    
                    # Extract specific SLA results
                    sla_results = json_results.get("sla_results", [])
                    for sla_result in sla_results:
                        if not sla_result.get("passed", True):
                            violation_details = sla_result.get("violation_details", "Unknown SLA violation")
                            performance_results["violations"].append(violation_details)
                    
                except (json.JSONDecodeError, KeyError) as e:
                    self.logger.warning(f"Failed to parse performance JSON results: {e}")
        
        # Parse violations from stdout if JSON not available
        if not performance_results["violations"] and result.stdout:
            performance_results["violations"] = self._extract_performance_violations_from_output(result.stdout)
        
        return performance_results

    def _extract_performance_violations_from_output(self, output: str) -> List[str]:
        """
        Extract performance violations from command output text.
        
        Args:
            output: Command output text
            
        Returns:
            List of extracted violation messages
        """
        violations = []
        
        # Look for common performance violation patterns
        lines = output.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['violation', 'failed', 'exceeds', 'sla']):
                if any(perf_keyword in line.lower() for perf_keyword in ['performance', 'time', 'speed', 'benchmark']):
                    violations.append(line.strip())
        
        return violations

    def _apply_performance_validation_rules(self, 
                                          category: QualityGateCategory, 
                                          results: Dict[str, Any]) -> None:
        """
        Apply performance validation rules per TST-PERF-001 and TST-PERF-002.
        
        Args:
            category: Performance quality gate category to update
            results: Performance validation results
        """
        sla_violations = results["sla_violations"]
        performance_regressions = results["performance_regressions"]
        
        # Add SLA violations to category
        if sla_violations > 0:
            category.violations.extend(results["violations"])
        
        # Check for performance regressions
        if performance_regressions > 0:
            violation = (
                f"Performance regressions detected: {performance_regressions} operations "
                "showing degraded performance compared to baseline"
            )
            category.violations.append(violation)
        
        # Store metadata for reporting
        category.metadata.update({
            "sla_violations": sla_violations,
            "performance_regressions": performance_regressions,
            "return_code": results["return_code"],
            "overall_status": results.get("overall_status", "UNKNOWN")
        })

    def validate_execution_quality_gate(self) -> QualityGateCategory:
        """
        Execute test execution quality gate validation implementing TST-INF-002 
        test execution reliability requirements.
        
        Returns:
            QualityGateCategory with test execution validation results
        """
        self.logger.info("Executing test execution quality gate validation")
        category_start_time = time.time()
        
        # Initialize execution quality gate category
        execution_category = QualityGateCategory(
            name="execution",
            weight=25.0,  # 25% weight in overall quality score
            critical=False,  # Execution warnings don't block deployment
            validation_rules=self.quality_gates_config["categories"]["execution_validation"]["validation_rules"]
        )
        
        try:
            # Validate test execution timeouts and reliability
            self._validate_test_execution_criteria(execution_category)
            
            # Execution quality gate is primarily informational
            execution_category.passed = len(execution_category.violations) == 0
            
            self.logger.info(
                f"Execution quality gate: {'PASSED' if execution_category.passed else 'WARNING'} "
                f"({len(execution_category.violations)} warnings)"
            )
            
        except Exception as e:
            execution_category.violations.append(f"Test execution validation failed: {e}")
            self.logger.error(f"Test execution validation error: {e}")
        
        # Record execution time
        execution_category.execution_time = time.time() - category_start_time
        
        return execution_category

    def _validate_test_execution_criteria(self, category: QualityGateCategory) -> None:
        """
        Validate test execution criteria per TST-INF-002 requirements.
        
        Args:
            category: Execution quality gate category to update
        """
        current_execution_time = time.time() - self.start_time
        
        # Validate overall execution time
        max_execution_time = self.quality_gates_config["execution"]["timeouts"]["test_execution_timeout"]
        if current_execution_time > max_execution_time:
            violation = (
                f"Quality gate execution time {current_execution_time:.1f}s exceeds "
                f"maximum {max_execution_time}s (TST-INF-002 violation)"
            )
            category.violations.append(violation)
        
        # Check for minimum test requirements (this would typically be validated 
        # as part of the coverage validation, but we include basic checks here)
        minimum_test_count = self.quality_gates_config["execution"]["requirements"]["minimum_test_count"]
        
        # This is a placeholder - in a real implementation, you would extract
        # test counts from pytest execution results
        estimated_test_count = 100  # Would be extracted from test results
        
        if estimated_test_count < minimum_test_count:
            warning = (
                f"Estimated test count {estimated_test_count} below recommended "
                f"minimum {minimum_test_count} tests"
            )
            category.warnings.append(warning)
        
        # Store execution metadata
        category.metadata.update({
            "execution_time": current_execution_time,
            "max_execution_time": max_execution_time,
            "estimated_test_count": estimated_test_count,
            "minimum_test_count": minimum_test_count
        })

    def execute_comprehensive_validation(self) -> QualityGateValidationResult:
        """
        Execute comprehensive quality gate validation across all categories.
        
        Implements complete validation orchestration per Section 4.1.1.5 test 
        execution workflow with automated merge blocking per TST-COV-004.
        
        Returns:
            QualityGateValidationResult with complete validation analysis
        """
        self.logger.info("=" * 80)
        self.logger.info("INITIATING COMPREHENSIVE QUALITY GATE VALIDATION")
        self.logger.info("=" * 80)
        
        validation_start_time = time.time()
        
        # Initialize result structure
        result = QualityGateValidationResult(
            overall_passed=False,
            quality_score=0.0,
            timestamp=datetime.now()
        )
        
        try:
            # Execute coverage quality gate validation
            self.logger.info("Phase 1: Coverage Quality Gate Validation")
            coverage_category = self.validate_coverage_quality_gate()
            result.categories["coverage"] = coverage_category
            
            # Execute performance quality gate validation
            self.logger.info("Phase 2: Performance Quality Gate Validation")
            performance_category = self.validate_performance_quality_gate()
            result.categories["performance"] = performance_category
            
            # Execute test execution quality gate validation
            self.logger.info("Phase 3: Test Execution Quality Gate Validation")
            execution_category = self.validate_execution_quality_gate()
            result.categories["execution"] = execution_category
            
            # Calculate overall quality metrics
            self._calculate_overall_quality_metrics(result)
            
            # Determine merge allowance per TST-COV-004
            self._determine_merge_allowance(result)
            
            # Generate comprehensive recommendations
            self._generate_quality_recommendations(result)
            
            # Record execution summary
            total_execution_time = time.time() - validation_start_time
            result.execution_summary = {
                "total_execution_time": total_execution_time,
                "categories_validated": len(result.categories),
                "scripts_executed": ["validate-coverage.py", "check-performance-slas.py"],
                "validation_timestamp": result.timestamp.isoformat(),
                "workspace_root": str(self.workspace_root)
            }
            
            self.logger.info(f"Quality gate validation completed in {total_execution_time:.2f} seconds")
            self.logger.info(f"Overall result: {'PASSED' if result.overall_passed else 'FAILED'}")
            self.logger.info(f"Merge allowed: {'YES' if result.merge_allowed else 'NO'}")
            
        except Exception as e:
            self.logger.error(f"Critical error during quality gate validation: {e}")
            result.critical_failures.append(f"Validation system failure: {e}")
            result.overall_passed = False
            result.merge_allowed = False
        
        # Store result for reporting
        self.validation_result = result
        
        return result

    def _calculate_overall_quality_metrics(self, result: QualityGateValidationResult) -> None:
        """
        Calculate overall quality metrics from category results.
        
        Args:
            result: Quality gate validation result to update
        """
        total_violations = 0
        total_warnings = 0
        weighted_score = 0.0
        total_weight = 0.0
        critical_failures = []
        
        for category_name, category in result.categories.items():
            # Count violations and warnings
            total_violations += len(category.violations)
            total_warnings += len(category.warnings)
            
            # Calculate weighted score
            category_score = 100.0 if category.passed else 0.0
            weighted_score += category_score * category.weight
            total_weight += category.weight
            
            # Collect critical failures
            if category.critical and not category.passed:
                critical_failures.extend(category.violations)
        
        # Calculate overall quality score
        result.quality_score = weighted_score / total_weight if total_weight > 0 else 0.0
        result.total_violations = total_violations
        result.total_warnings = total_warnings
        result.critical_failures = critical_failures
        
        # Determine overall pass status
        all_critical_passed = all(
            category.passed for category in result.categories.values() 
            if category.critical
        )
        result.overall_passed = all_critical_passed and len(critical_failures) == 0

    def _determine_merge_allowance(self, result: QualityGateValidationResult) -> None:
        """
        Determine merge allowance per TST-COV-004 automated merge blocking.
        
        Args:
            result: Quality gate validation result to update
        """
        # Check global quality gate decision logic
        global_config = self.quality_gates_config.get("global", {})
        decision_config = global_config.get("quality_gate_decision", {})
        
        require_all_categories = decision_config.get("require_all_categories_pass", True)
        allow_exemptions = decision_config.get("allow_category_exemptions", False)
        emergency_bypass = decision_config.get("emergency_bypass_enabled", False)
        
        # Determine merge allowance based on configuration
        if require_all_categories:
            # All categories must pass
            result.merge_allowed = result.overall_passed
        else:
            # Only critical categories must pass
            critical_categories_passed = all(
                category.passed for category in result.categories.values() 
                if category.critical
            )
            result.merge_allowed = critical_categories_passed
        
        # Apply emergency bypass if configured
        if emergency_bypass and not result.merge_allowed:
            self.logger.warning("Emergency bypass enabled - merge allowed despite quality gate failures")
            result.merge_allowed = True
        
        # Log merge decision reasoning
        if result.merge_allowed:
            self.logger.info("✅ MERGE ALLOWED - All quality gates satisfied")
        else:
            self.logger.error("❌ MERGE BLOCKED - Quality gate violations detected")
            for failure in result.critical_failures:
                self.logger.error(f"  Critical failure: {failure}")

    def _generate_quality_recommendations(self, result: QualityGateValidationResult) -> None:
        """
        Generate actionable quality improvement recommendations.
        
        Args:
            result: Quality gate validation result to update
        """
        recommendations = []
        
        # Coverage-specific recommendations
        if "coverage" in result.categories and not result.categories["coverage"].passed:
            recommendations.extend([
                "Add comprehensive unit tests for untested code paths",
                "Focus on critical modules requiring 100% coverage per TST-COV-002",
                "Review test fixtures to ensure proper edge case coverage",
                "Consider property-based testing for complex validation logic"
            ])
        
        # Performance-specific recommendations
        if "performance" in result.categories and not result.categories["performance"].passed:
            recommendations.extend([
                "Profile data loading operations for bottleneck identification",
                "Optimize DataFrame transformation algorithms for large datasets",
                "Consider caching strategies for frequently accessed data",
                "Review memory usage patterns to prevent performance degradation"
            ])
        
        # Execution-specific recommendations
        if "execution" in result.categories and not result.categories["execution"].passed:
            recommendations.extend([
                "Optimize test execution times through better fixture management",
                "Consider parallel test execution for faster CI/CD cycles",
                "Review test isolation to prevent cascading failures",
                "Implement test result caching for unchanged code paths"
            ])
        
        # General quality recommendations
        if not result.overall_passed:
            recommendations.extend([
                "Execute all quality gate validations locally before committing",
                "Monitor quality metrics trends to prevent regression accumulation",
                "Consider pair programming for complex feature development",
                "Establish regular quality gate threshold reviews with stakeholders"
            ])
        
        result.recommendations = recommendations

    def generate_comprehensive_report(self, 
                                    output_format: str = "console",
                                    output_path: Optional[Path] = None) -> str:
        """
        Generate comprehensive quality gate validation report.
        
        Args:
            output_format: Report format ("console", "json", "html")
            output_path: Optional output file path
            
        Returns:
            Generated report content or file path
        """
        if not self.validation_result:
            raise ValueError("No validation result available - execute validation first")
        
        if output_format == "console":
            return self._generate_console_report()
        elif output_format == "json":
            return self._generate_json_report(output_path)
        elif output_format == "html":
            return self._generate_html_report(output_path)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _generate_console_report(self) -> str:
        """
        Generate comprehensive console report for quality gate validation.
        
        Returns:
            Formatted console report string
        """
        result = self.validation_result
        report_lines = [
            "=" * 80,
            "FLYRIGLOADER QUALITY GATE VALIDATION REPORT",
            "=" * 80,
            f"Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Execution Time: {result.execution_summary['total_execution_time']:.2f} seconds",
            f"Workspace: {result.execution_summary['workspace_root']}",
            "",
            "OVERALL QUALITY GATE STATUS",
            "-" * 40,
        ]
        
        # Overall status with prominent indicators
        status_indicator = "✅ PASSED" if result.overall_passed else "❌ FAILED"
        merge_indicator = "✅ ALLOWED" if result.merge_allowed else "❌ BLOCKED"
        
        report_lines.extend([
            f"Overall Status: {status_indicator}",
            f"Quality Score: {result.quality_score:.1f}/100.0",
            f"Merge Status: {merge_indicator}",
            f"Total Violations: {result.total_violations}",
            f"Total Warnings: {result.total_warnings}",
            ""
        ])
        
        # Category breakdown
        report_lines.extend([
            "QUALITY GATE CATEGORY RESULTS",
            "-" * 40,
        ])
        
        for category_name, category in result.categories.items():
            status_icon = "✅" if category.passed else ("❌" if category.critical else "⚠️")
            weight_info = f" (weight: {category.weight}%)" if category.weight > 0 else ""
            critical_info = " [CRITICAL]" if category.critical else ""
            
            report_lines.extend([
                f"{status_icon} {category_name.upper()}{critical_info}{weight_info}",
                f"  Execution Time: {category.execution_time:.2f}s",
                f"  Violations: {len(category.violations)}",
                f"  Warnings: {len(category.warnings)}"
            ])
            
            # Add violation details
            if category.violations:
                report_lines.append("  Violation Details:")
                for violation in category.violations:
                    report_lines.append(f"    • {violation}")
            
            # Add warning details
            if category.warnings:
                report_lines.append("  Warnings:")
                for warning in category.warnings:
                    report_lines.append(f"    • {warning}")
            
            report_lines.append("")
        
        # Critical failures section
        if result.critical_failures:
            report_lines.extend([
                "CRITICAL FAILURES (MERGE BLOCKING)",
                "-" * 40,
            ])
            for i, failure in enumerate(result.critical_failures, 1):
                report_lines.append(f"{i}. {failure}")
            report_lines.append("")
        
        # Recommendations section
        if result.recommendations:
            report_lines.extend([
                "ACTIONABLE RECOMMENDATIONS",
                "-" * 40,
            ])
            for i, recommendation in enumerate(result.recommendations, 1):
                report_lines.append(f"{i}. {recommendation}")
            report_lines.append("")
        
        # Quality gate configuration summary
        report_lines.extend([
            "CONFIGURATION SUMMARY",
            "-" * 40,
            f"Quality Gates Config: {self.quality_gates_config_path}",
            f"Coverage Config: {self.coverage_config_path}",
            f"Coverage Threshold: {self.quality_gates_config['coverage']['overall_coverage_threshold']}%",
            f"Critical Module Threshold: {self.quality_gates_config['coverage']['critical_module_coverage_threshold']}%",
            ""
        ])
        
        # Footer
        report_lines.extend([
            "=" * 80,
            "Quality gate enforcement completed",
            f"For detailed analysis: {self.workspace_root}/tests/coverage/logs/",
            "For support: https://github.com/flyrigloader/flyrigloader/issues",
            "=" * 80
        ])
        
        report_content = "\n".join(report_lines)
        print(report_content)
        return report_content

    def _generate_json_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate machine-readable JSON report for CI/CD integration.
        
        Args:
            output_path: Optional output file path
            
        Returns:
            Path to generated JSON report file
        """
        result = self.validation_result
        
        # Convert result to JSON-serializable format
        report_data = {
            "timestamp": result.timestamp.isoformat(),
            "overall_status": {
                "passed": result.overall_passed,
                "quality_score": result.quality_score,
                "merge_allowed": result.merge_allowed,
                "total_violations": result.total_violations,
                "total_warnings": result.total_warnings
            },
            "categories": {},
            "critical_failures": result.critical_failures,
            "recommendations": result.recommendations,
            "execution_summary": result.execution_summary,
            "configuration": {
                "quality_gates_config": str(self.quality_gates_config_path),
                "coverage_config": str(self.coverage_config_path),
                "workspace_root": str(self.workspace_root)
            }
        }
        
        # Add category details
        for category_name, category in result.categories.items():
            report_data["categories"][category_name] = {
                "name": category.name,
                "weight": category.weight,
                "critical": category.critical,
                "passed": category.passed,
                "violations": category.violations,
                "warnings": category.warnings,
                "execution_time": category.execution_time,
                "metadata": category.metadata
            }
        
        # Determine output path
        if output_path is None:
            timestamp_str = result.timestamp.strftime("%Y%m%d_%H%M%S")
            output_path = self.workspace_root / f"quality-gate-report_{timestamp_str}.json"
        
        # Write JSON report
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"JSON report generated: {output_path}")
        return str(output_path)

    def _generate_html_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate comprehensive HTML report for detailed analysis.
        
        Args:
            output_path: Optional output file path
            
        Returns:
            Path to generated HTML report file
        """
        result = self.validation_result
        
        # Determine output path
        if output_path is None:
            timestamp_str = result.timestamp.strftime("%Y%m%d_%H%M%S")
            output_path = self.workspace_root / f"quality-gate-report_{timestamp_str}.html"
        
        # Generate HTML content
        html_content = self._create_html_report_content()
        
        # Write HTML report
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report generated: {output_path}")
        return str(output_path)

    def _create_html_report_content(self) -> str:
        """
        Create comprehensive HTML content for quality gate report.
        
        Returns:
            Complete HTML report content
        """
        result = self.validation_result
        
        # Determine status colors
        overall_color = "green" if result.overall_passed else "red"
        merge_color = "green" if result.merge_allowed else "red"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>FlyRigLoader Quality Gate Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .status-passed {{ color: green; font-weight: bold; }}
        .status-failed {{ color: red; font-weight: bold; }}
        .status-warning {{ color: orange; font-weight: bold; }}
        .category {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .category.passed {{ border-color: green; background-color: #f0f8ff; }}
        .category.failed {{ border-color: red; background-color: #fff0f0; }}
        .category.warning {{ border-color: orange; background-color: #fffbf0; }}
        .metadata {{ background-color: #f9f9f9; padding: 10px; border-radius: 3px; margin: 10px 0; }}
        .recommendations {{ background-color: #e8f4fd; padding: 15px; border-radius: 5px; }}
        .critical-failures {{ background-color: #ffeaa7; padding: 15px; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
        .violation {{ color: #d63031; margin: 5px 0; }}
        .warning {{ color: #fdcb6e; margin: 5px 0; }}
        .score-bar {{ background-color: #ddd; height: 20px; border-radius: 10px; overflow: hidden; }}
        .score-fill {{ height: 100%; background-color: {overall_color}; transition: width 0.3s ease; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>FlyRigLoader Quality Gate Validation Report</h1>
        <p><strong>Generated:</strong> {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Execution Time:</strong> {result.execution_summary['total_execution_time']:.2f} seconds</p>
        <p><strong>Overall Status:</strong> <span class="status-{'passed' if result.overall_passed else 'failed'}">
           {'PASSED' if result.overall_passed else 'FAILED'}</span></p>
        <p><strong>Merge Status:</strong> <span style="color: {merge_color}; font-weight: bold;">
           {'ALLOWED' if result.merge_allowed else 'BLOCKED'}</span></p>
        <div class="score-bar">
            <div class="score-fill" style="width: {result.quality_score}%"></div>
        </div>
        <p><strong>Quality Score:</strong> {result.quality_score:.1f}/100.0</p>
    </div>
    
    <h2>Quality Gate Summary</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
            <th>Status</th>
        </tr>
        <tr>
            <td>Total Violations</td>
            <td>{result.total_violations}</td>
            <td><span class="{'status-passed' if result.total_violations == 0 else 'status-failed'}">
                {'✅' if result.total_violations == 0 else '❌'}</span></td>
        </tr>
        <tr>
            <td>Total Warnings</td>
            <td>{result.total_warnings}</td>
            <td><span class="{'status-passed' if result.total_warnings == 0 else 'status-warning'}">
                {'✅' if result.total_warnings == 0 else '⚠️'}</span></td>
        </tr>
        <tr>
            <td>Critical Failures</td>
            <td>{len(result.critical_failures)}</td>
            <td><span class="{'status-passed' if len(result.critical_failures) == 0 else 'status-failed'}">
                {'✅' if len(result.critical_failures) == 0 else '❌'}</span></td>
        </tr>
    </table>
"""
        
        # Add category results
        html_content += "<h2>Quality Gate Category Results</h2>"
        
        for category_name, category in result.categories.items():
            status_class = "passed" if category.passed else ("failed" if category.critical else "warning")
            status_icon = "✅" if category.passed else ("❌" if category.critical else "⚠️")
            critical_badge = '<span style="background-color: red; color: white; padding: 2px 5px; border-radius: 3px; font-size: 0.8em;">CRITICAL</span>' if category.critical else ''
            
            html_content += f"""
    <div class="category {status_class}">
        <h3>{status_icon} {category_name.upper()} {critical_badge}</h3>
        <p><strong>Weight:</strong> {category.weight}% | <strong>Execution Time:</strong> {category.execution_time:.2f}s</p>
        
        <div class="metadata">
            <h4>Category Metadata</h4>
            <ul>
"""
            
            # Add metadata details
            for key, value in category.metadata.items():
                html_content += f"<li><strong>{key.replace('_', ' ').title()}:</strong> {value}</li>"
            
            html_content += "</ul></div>"
            
            # Add violations
            if category.violations:
                html_content += "<h4>Violations</h4><ul>"
                for violation in category.violations:
                    html_content += f'<li class="violation">❌ {violation}</li>'
                html_content += "</ul>"
            
            # Add warnings
            if category.warnings:
                html_content += "<h4>Warnings</h4><ul>"
                for warning in category.warnings:
                    html_content += f'<li class="warning">⚠️ {warning}</li>'
                html_content += "</ul>"
            
            html_content += "</div>"
        
        # Add critical failures section
        if result.critical_failures:
            html_content += """
    <div class="critical-failures">
        <h2>Critical Failures (Merge Blocking)</h2>
        <p>The following critical failures prevent code merge per TST-COV-004 requirements:</p>
        <ul>
"""
            for failure in result.critical_failures:
                html_content += f"<li>❌ {failure}</li>"
            
            html_content += "</ul></div>"
        
        # Add recommendations section
        if result.recommendations:
            html_content += """
    <div class="recommendations">
        <h2>Actionable Recommendations</h2>
        <p>Implement the following recommendations to improve code quality:</p>
        <ol>
"""
            for recommendation in result.recommendations:
                html_content += f"<li>{recommendation}</li>"
            
            html_content += "</ol></div>"
        
        # Add configuration details
        html_content += f"""
    <h2>Configuration Details</h2>
    <table>
        <tr><th>Setting</th><th>Value</th></tr>
        <tr><td>Quality Gates Config</td><td>{self.quality_gates_config_path}</td></tr>
        <tr><td>Coverage Config</td><td>{self.coverage_config_path}</td></tr>
        <tr><td>Workspace Root</td><td>{self.workspace_root}</td></tr>
        <tr><td>Coverage Threshold</td><td>{self.quality_gates_config['coverage']['overall_coverage_threshold']}%</td></tr>
        <tr><td>Critical Module Threshold</td><td>{self.quality_gates_config['coverage']['critical_module_coverage_threshold']}%</td></tr>
    </table>
    
    <footer style="margin-top: 40px; padding: 20px; background-color: #f0f0f0; border-radius: 5px;">
        <p><strong>Report generated by FlyRigLoader Quality Gate Enforcement System</strong></p>
        <p>For support and documentation: <a href="https://github.com/flyrigloader/flyrigloader">https://github.com/flyrigloader/flyrigloader</a></p>
        <p>Quality gate configuration version: {self.quality_gates_config.get('metadata', {}).get('version', 'unknown')}</p>
    </footer>
</body>
</html>
"""
        
        return html_content


# ============================================================================
# COMMAND LINE INTERFACE AND MAIN EXECUTION
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for quality gate enforcement script.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Comprehensive Quality Gate Enforcement with CI/CD Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enforce-quality-gates.py
  python enforce-quality-gates.py --fail-on-violation
  python enforce-quality-gates.py --output-format json --output-path report.json
  python enforce-quality-gates.py --ci-mode --verbose

CI/CD Integration:
  Exit code 0: All quality gates passed, merge allowed
  Exit code 1: Quality gate violations detected, merge blocked
  Exit code 2: System failure during validation
        """
    )
    
    parser.add_argument(
        "--quality-gates-config",
        type=Path,
        default=Path("tests/coverage/quality-gates.yml"),
        help="Path to quality gates configuration file"
    )
    
    parser.add_argument(
        "--coverage-config",
        type=Path,
        default=Path("tests/coverage/coverage-thresholds.json"),
        help="Path to coverage thresholds configuration file"
    )
    
    parser.add_argument(
        "--workspace-root",
        type=Path,
        help="Root workspace directory (defaults to current directory)"
    )
    
    parser.add_argument(
        "--output-format",
        choices=["console", "json", "html"],
        default="console",
        help="Output format for quality gate report"
    )
    
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Output path for generated report file"
    )
    
    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        default=True,
        help="Exit with code 1 on quality gate violations (default: True)"
    )
    
    parser.add_argument(
        "--ci-mode",
        action="store_true",
        help="Enable CI/CD mode with structured logging and status outputs"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform validation without executing actual quality gate checks"
    )
    
    return parser.parse_args()


def main() -> int:
    """
    Main entry point for quality gate enforcement script implementing comprehensive
    validation orchestration per TST-COV-004 and Section 4.1.1.5 requirements.
    
    Returns:
        Exit code: 0 for success, 1 for violations, 2 for system failure
    """
    args = parse_arguments()
    
    # Configure logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize quality gate enforcer
        enforcer = QualityGateEnforcer(
            quality_gates_config_path=args.quality_gates_config,
            coverage_config_path=args.coverage_config,
            workspace_root=args.workspace_root
        )
        
        # Load configurations
        enforcer.load_configurations()
        
        # Execute comprehensive validation (unless dry run)
        if args.dry_run:
            enforcer.logger.info("DRY RUN MODE: Configuration validation completed successfully")
            print("✅ Dry run completed - configurations are valid")
            return 0
        
        # Execute full quality gate validation
        result = enforcer.execute_comprehensive_validation()
        
        # Generate and display report
        report_output = enforcer.generate_comprehensive_report(
            output_format=args.output_format,
            output_path=args.output_path
        )
        
        # CI/CD mode structured outputs
        if args.ci_mode:
            print(f"::set-output name=quality_gate_passed::{result.overall_passed}")
            print(f"::set-output name=merge_allowed::{result.merge_allowed}")
            print(f"::set-output name=quality_score::{result.quality_score:.2f}")
            print(f"::set-output name=total_violations::{result.total_violations}")
            print(f"::set-output name=execution_time::{result.execution_summary['total_execution_time']:.2f}")
            
            if args.output_format != "console":
                print(f"::set-output name=report_path::{report_output}")
        
        # Determine exit code based on validation results and configuration
        if result.overall_passed and result.merge_allowed:
            enforcer.logger.info("✅ All quality gates passed - merge allowed")
            return 0
        elif args.fail_on_violation:
            enforcer.logger.error("❌ Quality gate violations detected - merge blocked per TST-COV-004")
            return 1
        else:
            enforcer.logger.warning("⚠️ Quality gate violations detected but failure disabled")
            return 0
            
    except FileNotFoundError as e:
        print(f"❌ Configuration file not found: {e}", file=sys.stderr)
        return 2
        
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        print(f"❌ Configuration file parsing error: {e}", file=sys.stderr)
        return 2
        
    except Exception as e:
        print(f"❌ Quality gate enforcement system failure: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())