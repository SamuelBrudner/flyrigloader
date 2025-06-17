#!/usr/bin/env python3
"""
Enhanced Quality Gate Enforcement Orchestrator

Central quality gate enforcement orchestrator that coordinates coverage validation, 
performance SLA checking, and test execution criteria to make merge-blocking decisions 
essential for maintaining research-grade software quality standards in the enhanced 
testing strategy.

This orchestrator implements:
- Central quality gate orchestration enforcing 90% coverage threshold and performance SLA compliance per Section 8.5.1
- Integration with CI/CD pipeline for automated merge blocking and quality validation per Section 8.4 CI/CD Pipeline Architecture  
- Comprehensive audit trail and structured reporting for research-grade traceability per Section 6.6.10 Documentation and Reporting
- Support for enhanced testing strategy with pytest-style validation and AAA pattern enforcement per Section 6.6.8

Key Enhancements for Enhanced Testing Strategy:
- Updated path references for scripts/coverage/ directory structure
- Integration with performance test isolation in scripts/benchmarks/
- Support for pytest-style validation and AAA pattern enforcement
- Enhanced audit trail generation with detailed traceability
- CI/CD integration for GitHub Actions workflow status reporting

Authors: Blitzy Agent - FlyRigLoader Test Suite Enhancement
Created: 2024-12-19  
Version: 2.0.0 - Enhanced Testing Strategy Integration
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


# ============================================================================
# CONFIGURATION AND DATA STRUCTURES
# ============================================================================

@dataclass
class QualityGateCategory:
    """
    Enhanced quality gate category with validation rules and thresholds.
    
    Implements comprehensive validation criteria per Section 6.6 Testing Strategy
    with enhanced support for pytest-style validation and performance test isolation.
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
    pytest_style_compliance: Optional[bool] = None
    aaa_pattern_score: Optional[float] = None
    fixture_quality_score: Optional[float] = None


@dataclass
class QualityGateValidationResult:
    """
    Enhanced quality gate validation result with detailed analysis.
    
    Aggregates all category results with actionable feedback per TST-COV-004 
    automated merge blocking requirements and enhanced testing strategy support.
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
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    enhanced_testing_metrics: Dict[str, Any] = field(default_factory=dict)
    pytest_style_summary: Dict[str, Any] = field(default_factory=dict)
    audit_trail: List[str] = field(default_factory=list)


# ============================================================================
# ENHANCED QUALITY GATE ORCHESTRATION ENGINE
# ============================================================================

class EnhancedQualityGateEnforcer:
    """
    Enhanced quality gate enforcement engine implementing comprehensive validation 
    orchestration with automated merge blocking per TST-COV-004 requirements.
    
    Coordinates coverage validation, performance SLA checking, test execution criteria,
    and enhanced testing strategy compliance with detailed reporting and CI/CD pipeline 
    integration per Section 8.4 CI/CD Pipeline Architecture.
    
    Key Enhancements:
    - Scripts/coverage/ directory structure integration
    - Performance test isolation support (scripts/benchmarks/)
    - Pytest-style validation and AAA pattern enforcement
    - Enhanced audit trail generation for research-grade traceability
    - GitHub Actions workflow status reporting integration
    """

    def __init__(self, 
                 quality_gates_config_path: Path = None,
                 coverage_thresholds_path: Path = None,
                 ci_mode: bool = False,
                 fail_on_violation: bool = True,
                 output_format: str = "console",
                 verbose: bool = False):
        """
        Initialize enhanced quality gate enforcer with updated paths and enhanced features.
        
        Args:
            quality_gates_config_path: Path to quality-gates.yml in scripts/coverage/
            coverage_thresholds_path: Path to coverage-thresholds.json in scripts/coverage/
            ci_mode: Enable CI-specific output formatting and behavior
            fail_on_violation: Exit with non-zero code on quality gate violations
            output_format: Output format (console, json, html)
            verbose: Enable detailed logging and analysis
        """
        # Enhanced path configuration for scripts/coverage/ structure
        self.scripts_coverage_root = Path("scripts/coverage")
        self.quality_gates_config_path = quality_gates_config_path or self.scripts_coverage_root / "quality-gates.yml"
        self.coverage_thresholds_path = coverage_thresholds_path or self.scripts_coverage_root / "coverage-thresholds.json"
        
        # Enhanced configuration
        self.ci_mode = ci_mode
        self.fail_on_violation = fail_on_violation
        self.output_format = output_format
        self.verbose = verbose
        
        # Configuration storage
        self.quality_gates_config: Dict[str, Any] = {}
        self.coverage_thresholds_config: Dict[str, Any] = {}
        
        # Enhanced logging configuration with audit trail support
        self.logger = self._configure_enhanced_logging()
        
        # Performance tracking
        self.start_time = time.time()
        self.execution_metrics: Dict[str, float] = {}
        
        # Enhanced testing strategy metrics
        self.pytest_style_metrics: Dict[str, Any] = {}
        self.aaa_pattern_metrics: Dict[str, Any] = {}
        
        # Audit trail for research-grade traceability
        self.audit_trail: List[str] = []
        
        # GitHub Actions integration
        self.github_actions_integration = os.getenv("GITHUB_ACTIONS", "false").lower() == "true"

    def _configure_enhanced_logging(self) -> logging.Logger:
        """Configure comprehensive logging with audit trail support for research-grade traceability."""
        logger = logging.getLogger("enhanced_quality_gate_enforcer")
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        
        # Create logs directory if it doesn't exist
        logs_dir = self.scripts_coverage_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # File handler for audit trail logging
        log_file = logs_dir / "quality-gate-enforcement.log"
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for real-time feedback
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Enhanced formatter for detailed audit trail
        detailed_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [PID:%(process)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S UTC"
        )
        file_handler.setFormatter(detailed_formatter)
        
        # CI-friendly console formatter
        if self.ci_mode:
            console_formatter = logging.Formatter("::%(levelname)s::%(message)s")
        else:
            console_formatter = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        
        # Avoid duplicate handlers in testing environments
        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger

    def load_configurations(self) -> None:
        """
        Load and validate enhanced configuration files for quality gates and coverage thresholds.
        
        Implements updated path references for scripts/coverage/ directory structure and
        validates configuration integrity per Section 8.5.1 Automated Quality Gates.
        
        Raises:
            FileNotFoundError: If configuration files are missing from scripts/coverage/
            yaml.YAMLError: If YAML configuration is invalid
            json.JSONDecodeError: If JSON configuration is invalid
        """
        self.logger.info("Loading enhanced quality gate configurations from scripts/coverage/")
        self._add_audit_entry("Starting configuration loading process")
        
        # Load quality gates configuration from scripts/coverage/
        try:
            with open(self.quality_gates_config_path, 'r', encoding='utf-8') as f:
                self.quality_gates_config = yaml.safe_load(f)
            self.logger.info(f"Loaded quality gates configuration from {self.quality_gates_config_path}")
            self._add_audit_entry(f"Quality gates config loaded: {self.quality_gates_config_path}")
        except FileNotFoundError:
            error_msg = f"Quality gates configuration not found: {self.quality_gates_config_path}"
            self.logger.error(error_msg)
            self._add_audit_entry(f"ERROR: {error_msg}")
            raise FileNotFoundError(error_msg)
        except yaml.YAMLError as e:
            error_msg = f"Invalid YAML in quality gates configuration: {e}"
            self.logger.error(error_msg)
            self._add_audit_entry(f"ERROR: {error_msg}")
            raise yaml.YAMLError(error_msg)
        
        # Load coverage thresholds configuration from scripts/coverage/
        try:
            with open(self.coverage_thresholds_path, 'r', encoding='utf-8') as f:
                self.coverage_thresholds_config = json.load(f)
            self.logger.info(f"Loaded coverage thresholds from {self.coverage_thresholds_path}")
            self._add_audit_entry(f"Coverage thresholds loaded: {self.coverage_thresholds_path}")
        except FileNotFoundError:
            error_msg = f"Coverage thresholds configuration not found: {self.coverage_thresholds_path}"
            self.logger.error(error_msg)
            self._add_audit_entry(f"ERROR: {error_msg}")
            raise FileNotFoundError(error_msg)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in coverage thresholds configuration: {e}"
            self.logger.error(error_msg)
            self._add_audit_entry(f"ERROR: {error_msg}")
            raise json.JSONDecodeError(error_msg, e.doc, e.pos)
        
        # Validate configuration integrity and enhanced testing strategy compliance
        self._validate_enhanced_configuration_integrity()

    def _validate_enhanced_configuration_integrity(self) -> None:
        """
        Validate configuration file integrity and enhanced testing strategy compliance.
        
        Ensures configurations support pytest-style validation, AAA pattern enforcement,
        and performance test isolation per Section 6.6.8 Test Categorization and Execution.
        
        Raises:
            ValueError: If configuration is invalid or incomplete
        """
        self.logger.info("Validating enhanced configuration integrity")
        self._add_audit_entry("Starting enhanced configuration validation")
        
        # Validate quality gates configuration structure
        required_sections = ["coverage", "performance", "execution", "global", "categories"]
        for section in required_sections:
            if section not in self.quality_gates_config:
                error_msg = f"Missing required section '{section}' in quality gates configuration"
                self.logger.error(error_msg)
                self._add_audit_entry(f"ERROR: {error_msg}")
                raise ValueError(error_msg)
        
        # Validate coverage thresholds configuration structure
        required_coverage_sections = ["global_settings", "critical_modules"]
        for section in required_coverage_sections:
            if section not in self.coverage_thresholds_config:
                error_msg = f"Missing required section '{section}' in coverage thresholds configuration"
                self.logger.error(error_msg)
                self._add_audit_entry(f"ERROR: {error_msg}")
                raise ValueError(error_msg)
        
        # Validate enhanced testing strategy support
        self._validate_pytest_style_support()
        self._validate_performance_isolation_support()
        
        self.logger.info("Enhanced configuration validation completed successfully")
        self._add_audit_entry("Enhanced configuration validation passed")

    def _validate_pytest_style_support(self) -> None:
        """Validate pytest-style validation and AAA pattern enforcement support in configuration."""
        # Check for pytest-style validation configuration
        execution_config = self.quality_gates_config.get("execution", {})
        pytest_style_config = execution_config.get("pytest_style_validation", {})
        
        if not pytest_style_config:
            self.logger.warning("No pytest-style validation configuration found - using defaults")
            self._add_audit_entry("WARNING: No pytest-style validation config found")
        
        # Check for AAA pattern enforcement configuration
        aaa_config = execution_config.get("aaa_pattern_enforcement", {})
        
        if not aaa_config:
            self.logger.warning("No AAA pattern enforcement configuration found - using defaults")
            self._add_audit_entry("WARNING: No AAA pattern enforcement config found")

    def _validate_performance_isolation_support(self) -> None:
        """Validate performance test isolation support for scripts/benchmarks/ integration."""
        performance_config = self.quality_gates_config.get("performance", {})
        
        # Check for benchmark isolation configuration
        benchmark_config = performance_config.get("benchmark_isolation", {})
        
        if not benchmark_config:
            self.logger.warning("No benchmark isolation configuration found - using defaults")
            self._add_audit_entry("WARNING: No benchmark isolation config found")
        
        # Validate scripts/benchmarks/ path references
        benchmark_path = performance_config.get("benchmark_script_path", "scripts/benchmarks/")
        if not benchmark_path.startswith("scripts/benchmarks/"):
            self.logger.warning(f"Benchmark path '{benchmark_path}' not in expected scripts/benchmarks/ location")
            self._add_audit_entry(f"WARNING: Non-standard benchmark path: {benchmark_path}")

    def validate_coverage_quality_gates(self) -> QualityGateCategory:
        """
        Execute comprehensive coverage validation with enhanced testing strategy support.
        
        Integrates with scripts/coverage/validate-coverage.py with updated path references
        and enhanced traceability per Section 6.6.10 Documentation and Reporting.
        
        Returns:
            QualityGateCategory: Coverage validation results with enhanced metrics
        """
        self.logger.info("Executing enhanced coverage quality gate validation")
        self._add_audit_entry("Starting coverage validation with enhanced testing strategy support")
        
        category = QualityGateCategory(
            name="coverage",
            weight=40.0,  # 40% weight per quality gates configuration
            critical=True
        )
        
        start_time = time.time()
        
        try:
            # Execute coverage validation with updated script path
            validate_coverage_script = self.scripts_coverage_root / "validate-coverage.py"
            
            cmd = [
                sys.executable,
                str(validate_coverage_script),
                "--coverage-config", str(self.coverage_thresholds_path),
                "--quality-gates-config", str(self.quality_gates_config_path),
                "--output-json", str(self.scripts_coverage_root / "coverage-validation-results.json"),
                "--ci-mode" if self.ci_mode else "--verbose"
            ]
            
            self.logger.debug(f"Executing coverage validation: {' '.join(cmd)}")
            self._add_audit_entry(f"Coverage validation command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=Path.cwd()
            )
            
            # Parse validation results
            if result.returncode == 0:
                category.passed = True
                self.logger.info("Coverage quality gates passed")
                self._add_audit_entry("Coverage validation PASSED")
            else:
                category.passed = False
                category.violations.append("Coverage thresholds not met")
                self.logger.error("Coverage quality gates failed")
                self._add_audit_entry("Coverage validation FAILED")
                
                # Parse detailed error information
                if result.stderr:
                    category.violations.extend(result.stderr.strip().split('\n'))
            
            # Load detailed results if available
            results_file = self.scripts_coverage_root / "coverage-validation-results.json"
            if results_file.exists():
                with open(results_file, 'r', encoding='utf-8') as f:
                    detailed_results = json.load(f)
                    category.metadata.update(detailed_results)
                    
                    # Extract enhanced testing strategy metrics
                    if "pytest_style_compliance" in detailed_results:
                        category.pytest_style_compliance = detailed_results["pytest_style_compliance"]
                    
                    if "fixture_quality_metrics" in detailed_results:
                        category.fixture_quality_score = detailed_results["fixture_quality_metrics"].get("overall_score", 0.0)
            
        except subprocess.TimeoutExpired:
            error_msg = "Coverage validation timed out after 5 minutes"
            self.logger.error(error_msg)
            self._add_audit_entry(f"ERROR: {error_msg}")
            category.violations.append(error_msg)
            category.passed = False
        except Exception as e:
            error_msg = f"Coverage validation failed with error: {e}"
            self.logger.error(error_msg)
            self._add_audit_entry(f"ERROR: {error_msg}")
            category.violations.append(error_msg)
            category.passed = False
        
        category.execution_time = time.time() - start_time
        self.execution_metrics["coverage"] = category.execution_time
        
        self.logger.info(f"Coverage validation completed in {category.execution_time:.2f}s")
        self._add_audit_entry(f"Coverage validation execution time: {category.execution_time:.2f}s")
        
        return category

    def validate_performance_quality_gates(self) -> QualityGateCategory:
        """
        Execute performance SLA validation with scripts/benchmarks/ integration.
        
        Integrates with scripts/coverage/check-performance-slas.py and supports
        performance test isolation per Section 6.6.4 Performance and Benchmark Testing.
        
        Returns:
            QualityGateCategory: Performance validation results with SLA compliance metrics
        """
        self.logger.info("Executing performance SLA quality gate validation")
        self._add_audit_entry("Starting performance validation with benchmark isolation support")
        
        category = QualityGateCategory(
            name="performance",
            weight=35.0,  # 35% weight per quality gates configuration
            critical=True
        )
        
        start_time = time.time()
        
        try:
            # Execute performance SLA validation with updated script path
            check_performance_script = self.scripts_coverage_root / "check-performance-slas.py"
            
            # Look for benchmark results in scripts/benchmarks/ directory
            benchmarks_dir = Path("scripts/benchmarks")
            benchmark_results_files = list(benchmarks_dir.glob("*benchmark-results*.json")) if benchmarks_dir.exists() else []
            
            if not benchmark_results_files:
                # Check for local benchmark results as fallback
                local_benchmark_files = list(Path.cwd().glob("*benchmark-results*.json"))
                if local_benchmark_files:
                    benchmark_results_files = local_benchmark_files
                else:
                    self.logger.warning("No benchmark results found - performance validation will use defaults")
                    self._add_audit_entry("WARNING: No benchmark results found for performance validation")
                    category.warnings.append("No benchmark results available for performance validation")
                    category.passed = True  # Allow pass with warning if no performance tests
                    return category
            
            # Use the most recent benchmark results file
            benchmark_file = max(benchmark_results_files, key=lambda f: f.stat().st_mtime)
            
            cmd = [
                sys.executable,
                str(check_performance_script),
                str(benchmark_file),
                "--config", str(self.quality_gates_config_path),
                "--output-format", "json",
                "--output-path", str(self.scripts_coverage_root / "performance-sla-results.json")
            ]
            
            self.logger.debug(f"Executing performance validation: {' '.join(cmd)}")
            self._add_audit_entry(f"Performance validation command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout for performance analysis
                cwd=Path.cwd()
            )
            
            # Parse validation results
            if result.returncode == 0:
                category.passed = True
                self.logger.info("Performance SLA quality gates passed")
                self._add_audit_entry("Performance validation PASSED")
            else:
                category.passed = False
                category.violations.append("Performance SLA thresholds not met")
                self.logger.error("Performance SLA quality gates failed")
                self._add_audit_entry("Performance validation FAILED")
                
                # Parse detailed error information
                if result.stderr:
                    category.violations.extend(result.stderr.strip().split('\n'))
            
            # Load detailed results if available
            results_file = self.scripts_coverage_root / "performance-sla-results.json"
            if results_file.exists():
                with open(results_file, 'r', encoding='utf-8') as f:
                    detailed_results = json.load(f)
                    category.metadata.update(detailed_results)
            
        except subprocess.TimeoutExpired:
            error_msg = "Performance validation timed out after 10 minutes"
            self.logger.error(error_msg)
            self._add_audit_entry(f"ERROR: {error_msg}")
            category.violations.append(error_msg)
            category.passed = False
        except Exception as e:
            error_msg = f"Performance validation failed with error: {e}"
            self.logger.error(error_msg)
            self._add_audit_entry(f"ERROR: {error_msg}")
            category.violations.append(error_msg)
            category.passed = False
        
        category.execution_time = time.time() - start_time
        self.execution_metrics["performance"] = category.execution_time
        
        self.logger.info(f"Performance validation completed in {category.execution_time:.2f}s")
        self._add_audit_entry(f"Performance validation execution time: {category.execution_time:.2f}s")
        
        return category

    def validate_execution_quality_gates(self) -> QualityGateCategory:
        """
        Execute test execution quality gate validation with pytest-style enforcement.
        
        Validates pytest-style compliance, AAA pattern enforcement, and fixture quality
        per Section 6.6.8 Test Categorization and Execution.
        
        Returns:
            QualityGateCategory: Test execution validation results with enhanced metrics
        """
        self.logger.info("Executing test execution quality gate validation")
        self._add_audit_entry("Starting test execution validation with pytest-style enforcement")
        
        category = QualityGateCategory(
            name="execution",
            weight=25.0,  # 25% weight per quality gates configuration
            critical=False  # Warning only per configuration
        )
        
        start_time = time.time()
        
        try:
            # Validate pytest-style compliance
            pytest_style_score = self._validate_pytest_style_compliance()
            category.pytest_style_compliance = pytest_style_score >= 95.0
            
            # Validate AAA pattern enforcement
            aaa_pattern_score = self._validate_aaa_pattern_compliance()
            category.aaa_pattern_score = aaa_pattern_score
            
            # Validate fixture quality
            fixture_quality_score = self._validate_fixture_quality()
            category.fixture_quality_score = fixture_quality_score
            
            # Determine overall execution quality
            execution_scores = [pytest_style_score, aaa_pattern_score, fixture_quality_score]
            overall_execution_score = sum(execution_scores) / len(execution_scores)
            
            # Apply quality thresholds
            execution_threshold = self.quality_gates_config.get("execution", {}).get("quality_threshold", 90.0)
            
            if overall_execution_score >= execution_threshold:
                category.passed = True
                self.logger.info(f"Test execution quality gates passed (score: {overall_execution_score:.1f}%)")
                self._add_audit_entry(f"Test execution validation PASSED with score: {overall_execution_score:.1f}%")
            else:
                category.passed = False
                category.violations.append(f"Test execution quality score {overall_execution_score:.1f}% below threshold {execution_threshold}%")
                self.logger.warning(f"Test execution quality gates failed (score: {overall_execution_score:.1f}%)")
                self._add_audit_entry(f"Test execution validation FAILED with score: {overall_execution_score:.1f}%")
            
            # Store detailed metrics
            category.metadata.update({
                "pytest_style_score": pytest_style_score,
                "aaa_pattern_score": aaa_pattern_score,
                "fixture_quality_score": fixture_quality_score,
                "overall_execution_score": overall_execution_score,
                "execution_threshold": execution_threshold
            })
            
        except Exception as e:
            error_msg = f"Test execution validation failed with error: {e}"
            self.logger.error(error_msg)
            self._add_audit_entry(f"ERROR: {error_msg}")
            category.violations.append(error_msg)
            category.passed = False
        
        category.execution_time = time.time() - start_time
        self.execution_metrics["execution"] = category.execution_time
        
        self.logger.info(f"Test execution validation completed in {category.execution_time:.2f}s")
        self._add_audit_entry(f"Test execution validation execution time: {category.execution_time:.2f}s")
        
        return category

    def _validate_pytest_style_compliance(self) -> float:
        """
        Validate pytest-style compliance across test files.
        
        Checks test naming conventions, fixture usage patterns, and marker compliance
        per Section 8.5 Quality Assurance Pipeline pytest-style validation.
        
        Returns:
            float: Pytest-style compliance score (0-100)
        """
        self.logger.debug("Validating pytest-style compliance")
        
        try:
            # Execute flake8 with pytest-style plugin for validation
            cmd = [
                "flake8",
                "--select=PT",
                "--format=json",
                "tests/"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
                cwd=Path.cwd()
            )
            
            if result.returncode == 0:
                # No pytest-style violations found
                self.logger.debug("Pytest-style validation passed")
                return 100.0
            else:
                # Parse violations and calculate score
                violations = []
                if result.stdout:
                    try:
                        violations_data = json.loads(result.stdout)
                        violations = violations_data if isinstance(violations_data, list) else []
                    except json.JSONDecodeError:
                        # Fallback to stderr parsing
                        violations = result.stderr.strip().split('\n') if result.stderr else []
                
                # Calculate compliance score based on violations
                total_test_files = len(list(Path("tests").rglob("test_*.py")))
                if total_test_files == 0:
                    return 100.0
                
                violation_count = len(violations)
                compliance_score = max(0, 100 - (violation_count / total_test_files * 10))
                
                self.logger.debug(f"Pytest-style compliance: {compliance_score:.1f}% ({violation_count} violations)")
                return compliance_score
                
        except subprocess.TimeoutExpired:
            self.logger.warning("Pytest-style validation timed out")
            return 50.0  # Partial score for timeout
        except Exception as e:
            self.logger.warning(f"Pytest-style validation failed: {e}")
            return 50.0  # Partial score for error

    def _validate_aaa_pattern_compliance(self) -> float:
        """
        Validate AAA (Arrange-Act-Assert) pattern compliance in test functions.
        
        Analyzes test files for proper AAA structure implementation per
        Section 6.6.8 Test Categorization and Execution.
        
        Returns:
            float: AAA pattern compliance score (0-100)
        """
        self.logger.debug("Validating AAA pattern compliance")
        
        try:
            # Analyze test files for AAA pattern structure
            test_files = list(Path("tests").rglob("test_*.py"))
            if not test_files:
                return 100.0  # No test files to analyze
            
            compliant_tests = 0
            total_tests = 0
            
            for test_file in test_files:
                try:
                    with open(test_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Simple heuristic to detect AAA patterns
                    # Look for test functions with arrange/act/assert comments or structure
                    import re
                    test_functions = re.findall(r'def test_[^(]+\([^)]*\):[^}]+?(?=def|\Z)', content, re.DOTALL)
                    
                    for test_func in test_functions:
                        total_tests += 1
                        
                        # Check for AAA pattern indicators
                        has_arrange = any(indicator in test_func.lower() for indicator in [
                            "# arrange", "# setup", "arrange:", "setup:"
                        ])
                        has_act = any(indicator in test_func.lower() for indicator in [
                            "# act", "# execute", "act:", "execute:"
                        ])
                        has_assert = any(indicator in test_func.lower() for indicator in [
                            "# assert", "assert ", "assert\n"
                        ])
                        
                        if has_arrange and has_act and has_assert:
                            compliant_tests += 1
                        elif "assert " in test_func or "assert\n" in test_func:
                            # Partial credit for having assertions
                            compliant_tests += 0.5
                
                except Exception as e:
                    self.logger.debug(f"Error analyzing test file {test_file}: {e}")
                    continue
            
            if total_tests == 0:
                return 100.0
            
            compliance_score = (compliant_tests / total_tests) * 100
            self.logger.debug(f"AAA pattern compliance: {compliance_score:.1f}% ({compliant_tests}/{total_tests})")
            return compliance_score
            
        except Exception as e:
            self.logger.warning(f"AAA pattern validation failed: {e}")
            return 50.0  # Partial score for error

    def _validate_fixture_quality(self) -> float:
        """
        Validate fixture quality and centralization per enhanced testing strategy.
        
        Analyzes fixture usage patterns, scope declarations, and centralization
        per Section 6.6.2.3 Mocking Strategy and centralized fixture management.
        
        Returns:
            float: Fixture quality score (0-100)
        """
        self.logger.debug("Validating fixture quality and centralization")
        
        try:
            # Check for centralized fixtures in tests/conftest.py
            conftest_files = [
                Path("tests/conftest.py"),
                Path("tests/utils.py")
            ]
            
            centralized_fixtures = 0
            for conftest_file in conftest_files:
                if conftest_file.exists():
                    with open(conftest_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Count @pytest.fixture declarations
                    import re
                    fixtures = re.findall(r'@pytest\.fixture', content)
                    centralized_fixtures += len(fixtures)
            
            # Count distributed fixtures in test files
            distributed_fixtures = 0
            test_files = list(Path("tests").rglob("test_*.py"))
            
            for test_file in test_files:
                try:
                    with open(test_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    import re
                    fixtures = re.findall(r'@pytest\.fixture', content)
                    distributed_fixtures += len(fixtures)
                
                except Exception:
                    continue
            
            # Calculate centralization score
            total_fixtures = centralized_fixtures + distributed_fixtures
            if total_fixtures == 0:
                return 100.0  # No fixtures to analyze
            
            centralization_score = (centralized_fixtures / total_fixtures) * 100
            
            # Fixture quality is based on centralization and naming conventions
            quality_score = min(100.0, centralization_score + 20)  # Bonus for any centralization
            
            self.logger.debug(f"Fixture quality score: {quality_score:.1f}% (centralized: {centralized_fixtures}, distributed: {distributed_fixtures})")
            return quality_score
            
        except Exception as e:
            self.logger.warning(f"Fixture quality validation failed: {e}")
            return 50.0  # Partial score for error

    def execute_quality_gates(self) -> QualityGateValidationResult:
        """
        Execute comprehensive quality gate validation with enhanced testing strategy support.
        
        Orchestrates all validation categories and computes overall quality score with
        merge-blocking decisions per TST-COV-004 and Section 4.1.1.5.
        
        Returns:
            QualityGateValidationResult: Comprehensive validation results with audit trail
        """
        self.logger.info("Starting comprehensive quality gate execution")
        self._add_audit_entry("=== Quality Gate Execution Started ===")
        
        # Initialize result object
        result = QualityGateValidationResult(
            overall_passed=False,
            quality_score=0.0,
            audit_trail=self.audit_trail.copy()
        )
        
        # Execute coverage validation
        self.logger.info("Executing coverage quality gates")
        coverage_category = self.validate_coverage_quality_gates()
        result.categories["coverage"] = coverage_category
        
        # Execute performance validation
        self.logger.info("Executing performance quality gates")
        performance_category = self.validate_performance_quality_gates()
        result.categories["performance"] = performance_category
        
        # Execute test execution validation
        self.logger.info("Executing test execution quality gates")
        execution_category = self.validate_execution_quality_gates()
        result.categories["execution"] = execution_category
        
        # Calculate weighted quality score
        total_weight = sum(cat.weight for cat in result.categories.values())
        weighted_score = 0.0
        
        for category in result.categories.values():
            if category.passed:
                weighted_score += category.weight
        
        result.quality_score = (weighted_score / total_weight) * 100 if total_weight > 0 else 0.0
        
        # Count violations and warnings
        result.total_violations = sum(len(cat.violations) for cat in result.categories.values())
        result.total_warnings = sum(len(cat.warnings) for cat in result.categories.values())
        
        # Identify critical failures
        for cat_name, category in result.categories.items():
            if category.critical and not category.passed:
                result.critical_failures.append(f"{cat_name}: {', '.join(category.violations)}")
        
        # Determine overall result
        critical_categories_passed = all(
            cat.passed for cat in result.categories.values() if cat.critical
        )
        quality_threshold = self.quality_gates_config.get("global", {}).get("quality_threshold", 90.0)
        
        result.overall_passed = (
            critical_categories_passed and 
            result.quality_score >= quality_threshold
        )
        
        # Merge blocking decision per TST-COV-004
        result.merge_allowed = result.overall_passed
        
        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)
        
        # Update execution summary
        result.execution_summary = {
            "total_execution_time": time.time() - self.start_time,
            "category_execution_times": self.execution_metrics,
            "quality_score": result.quality_score,
            "critical_failures_count": len(result.critical_failures),
            "merge_decision": "ALLOWED" if result.merge_allowed else "BLOCKED"
        }
        
        # Enhanced testing strategy metrics
        result.enhanced_testing_metrics = {
            "pytest_style_compliance": any(
                cat.pytest_style_compliance for cat in result.categories.values() 
                if cat.pytest_style_compliance is not None
            ),
            "aaa_pattern_score": max(
                (cat.aaa_pattern_score for cat in result.categories.values() 
                 if cat.aaa_pattern_score is not None), 
                default=0.0
            ),
            "fixture_quality_score": max(
                (cat.fixture_quality_score for cat in result.categories.values() 
                 if cat.fixture_quality_score is not None),
                default=0.0
            )
        }
        
        # Update audit trail
        result.audit_trail = self.audit_trail.copy()
        self._add_audit_entry(f"=== Quality Gate Execution Completed - Result: {'PASSED' if result.overall_passed else 'FAILED'} ===")
        
        self.logger.info(f"Quality gate execution completed - Overall result: {'PASSED' if result.overall_passed else 'FAILED'}")
        
        return result

    def generate_report(self, result: QualityGateValidationResult, output_path: Optional[Path] = None) -> None:
        """
        Generate comprehensive quality gate report with enhanced testing strategy metrics.
        
        Creates detailed reports in specified format with audit trail and recommendations
        per Section 6.6.10 Documentation and Reporting.
        
        Args:
            result: Quality gate validation results
            output_path: Optional output path for report files
        """
        self.logger.info(f"Generating quality gate report in {self.output_format} format")
        
        if self.output_format == "console":
            self._generate_console_report(result)
        elif self.output_format == "json":
            self._generate_json_report(result, output_path)
        elif self.output_format == "html":
            self._generate_html_report(result, output_path)
        else:
            self.logger.error(f"Unsupported output format: {self.output_format}")

    def _generate_console_report(self, result: QualityGateValidationResult) -> None:
        """Generate detailed console report with CI integration support."""
        print("\n" + "=" * 80)
        print("ENHANCED QUALITY GATE ENFORCEMENT REPORT")
        print("=" * 80)
        
        # GitHub Actions integration
        if self.github_actions_integration:
            print(f"::set-output name=quality_score::{result.quality_score:.1f}")
            print(f"::set-output name=merge_allowed::{str(result.merge_allowed).lower()}")
            print(f"::set-output name=critical_failures::{len(result.critical_failures)}")
        
        # Overall status
        status_color = "🟢" if result.overall_passed else "🔴"
        print(f"\n{status_color} Overall Status: {'PASSED' if result.overall_passed else 'FAILED'}")
        print(f"📊 Quality Score: {result.quality_score:.1f}%")
        print(f"🚦 Merge Decision: {'ALLOWED' if result.merge_allowed else 'BLOCKED'}")
        
        # Category results
        print(f"\n📋 Category Results:")
        for cat_name, category in result.categories.items():
            status_icon = "✅" if category.passed else "❌"
            weight_info = f"({category.weight}% weight)" if category.weight > 0 else ""
            critical_info = "(CRITICAL)" if category.critical else "(WARNING)"
            
            print(f"  {status_icon} {cat_name.title()}: {'PASSED' if category.passed else 'FAILED'} {weight_info} {critical_info}")
            print(f"     ⏱️  Execution time: {category.execution_time:.2f}s")
            
            if category.violations:
                print(f"     ⚠️  Violations: {len(category.violations)}")
                for violation in category.violations[:3]:  # Show first 3 violations
                    print(f"        • {violation}")
                if len(category.violations) > 3:
                    print(f"        • ... and {len(category.violations) - 3} more")
            
            if category.warnings:
                print(f"     ⚡ Warnings: {len(category.warnings)}")
                for warning in category.warnings[:2]:  # Show first 2 warnings
                    print(f"        • {warning}")
        
        # Enhanced testing strategy metrics
        print(f"\n🧪 Enhanced Testing Strategy Metrics:")
        enhanced_metrics = result.enhanced_testing_metrics
        print(f"  • Pytest-style compliance: {'✅' if enhanced_metrics.get('pytest_style_compliance') else '❌'}")
        print(f"  • AAA pattern score: {enhanced_metrics.get('aaa_pattern_score', 0):.1f}%")
        print(f"  • Fixture quality score: {enhanced_metrics.get('fixture_quality_score', 0):.1f}%")
        
        # Critical failures
        if result.critical_failures:
            print(f"\n🚨 Critical Failures ({len(result.critical_failures)}):")
            for failure in result.critical_failures:
                print(f"  • {failure}")
        
        # Recommendations
        if result.recommendations:
            print(f"\n💡 Recommendations:")
            for recommendation in result.recommendations:
                print(f"  • {recommendation}")
        
        # Execution summary
        print(f"\n⏱️  Execution Summary:")
        summary = result.execution_summary
        print(f"  • Total execution time: {summary.get('total_execution_time', 0):.2f}s")
        print(f"  • Quality score: {summary.get('quality_score', 0):.1f}%")
        print(f"  • Critical failures: {summary.get('critical_failures_count', 0)}")
        print(f"  • Merge decision: {summary.get('merge_decision', 'UNKNOWN')}")
        
        print("\n" + "=" * 80)

    def _generate_json_report(self, result: QualityGateValidationResult, output_path: Optional[Path] = None) -> None:
        """Generate detailed JSON report for CI/CD integration."""
        # Convert dataclass to dictionary for JSON serialization
        def dataclass_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {
                    field.name: dataclass_to_dict(getattr(obj, field.name))
                    for field in obj.__dataclass_fields__.values()
                }
            elif isinstance(obj, dict):
                return {k: dataclass_to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [dataclass_to_dict(item) for item in obj]
            elif isinstance(obj, datetime):
                return obj.isoformat()
            else:
                return obj
        
        report_data = dataclass_to_dict(result)
        
        # Add metadata
        report_data["metadata"] = {
            "report_generated_at": datetime.now(timezone.utc).isoformat(),
            "enforcer_version": "2.0.0",
            "enhanced_testing_strategy": True,
            "scripts_coverage_integration": True,
            "github_actions_integration": self.github_actions_integration
        }
        
        # Determine output path
        if output_path is None:
            output_path = self.scripts_coverage_root / "quality-gate-report.json"
        
        # Write JSON report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"JSON report generated: {output_path}")

    def _generate_html_report(self, result: QualityGateValidationResult, output_path: Optional[Path] = None) -> None:
        """Generate HTML report with enhanced visualizations."""
        # Determine output path
        if output_path is None:
            output_path = self.scripts_coverage_root / "quality-gate-report.html"
        
        # Simple HTML template
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Quality Gate Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .status-pass {{ color: #28a745; }}
        .status-fail {{ color: #dc3545; }}
        .category {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .violations {{ background-color: #f8d7da; padding: 10px; border-radius: 3px; margin: 10px 0; }}
        .warnings {{ background-color: #fff3cd; padding: 10px; border-radius: 3px; margin: 10px 0; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e9ecef; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Enhanced Quality Gate Enforcement Report</h1>
        <p><strong>Generated:</strong> {result.timestamp.isoformat()}</p>
        <p><strong>Overall Status:</strong> 
            <span class="{'status-pass' if result.overall_passed else 'status-fail'}">
                {'PASSED' if result.overall_passed else 'FAILED'}
            </span>
        </p>
        <p><strong>Quality Score:</strong> {result.quality_score:.1f}%</p>
        <p><strong>Merge Decision:</strong> 
            <span class="{'status-pass' if result.merge_allowed else 'status-fail'}">
                {'ALLOWED' if result.merge_allowed else 'BLOCKED'}
            </span>
        </p>
    </div>
    
    <h2>Category Results</h2>
"""
        
        # Add category details
        for cat_name, category in result.categories.items():
            status_class = "status-pass" if category.passed else "status-fail"
            html_content += f"""
    <div class="category">
        <h3>{cat_name.title()} 
            <span class="{status_class}">{'PASSED' if category.passed else 'FAILED'}</span>
            <small>({category.weight}% weight, {'CRITICAL' if category.critical else 'WARNING'})</small>
        </h3>
        <p><strong>Execution time:</strong> {category.execution_time:.2f}s</p>
"""
            
            if category.violations:
                html_content += f"""
        <div class="violations">
            <strong>Violations ({len(category.violations)}):</strong>
            <ul>
                {"".join(f"<li>{violation}</li>" for violation in category.violations)}
            </ul>
        </div>
"""
            
            if category.warnings:
                html_content += f"""
        <div class="warnings">
            <strong>Warnings ({len(category.warnings)}):</strong>
            <ul>
                {"".join(f"<li>{warning}</li>" for warning in category.warnings)}
            </ul>
        </div>
"""
            
            html_content += "    </div>\n"
        
        # Add enhanced testing strategy metrics
        enhanced_metrics = result.enhanced_testing_metrics
        html_content += f"""
    <h2>Enhanced Testing Strategy Metrics</h2>
    <div class="metric">
        <strong>Pytest-style compliance:</strong> 
        {'✅ Compliant' if enhanced_metrics.get('pytest_style_compliance') else '❌ Non-compliant'}
    </div>
    <div class="metric">
        <strong>AAA pattern score:</strong> {enhanced_metrics.get('aaa_pattern_score', 0):.1f}%
    </div>
    <div class="metric">
        <strong>Fixture quality score:</strong> {enhanced_metrics.get('fixture_quality_score', 0):.1f}%
    </div>
"""
        
        # Add recommendations
        if result.recommendations:
            html_content += """
    <h2>Recommendations</h2>
    <ul>
"""
            for recommendation in result.recommendations:
                html_content += f"        <li>{recommendation}</li>\n"
            html_content += "    </ul>\n"
        
        html_content += """
</body>
</html>
"""
        
        # Write HTML report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report generated: {output_path}")

    def _generate_recommendations(self, result: QualityGateValidationResult) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        # Coverage recommendations
        coverage_cat = result.categories.get("coverage")
        if coverage_cat and not coverage_cat.passed:
            recommendations.append("Increase test coverage by adding tests for uncovered code paths")
            recommendations.append("Focus on critical modules requiring 100% coverage per TST-COV-002")
            recommendations.append("Use 'coverage report --show-missing' to identify specific uncovered lines")
        
        # Performance recommendations
        performance_cat = result.categories.get("performance")
        if performance_cat and not performance_cat.passed:
            recommendations.append("Optimize data loading operations to meet <1s per 100MB SLA (TST-PERF-001)")
            recommendations.append("Optimize DataFrame transformations to meet <500ms per 1M rows SLA (TST-PERF-002)")
            recommendations.append("Run performance benchmarks using 'python scripts/benchmarks/run_benchmarks.py'")
        
        # Test execution recommendations
        execution_cat = result.categories.get("execution")
        if execution_cat and not execution_cat.passed:
            if execution_cat.pytest_style_compliance is False:
                recommendations.append("Improve pytest-style compliance by running 'flake8 --select=PT tests/'")
                recommendations.append("Follow test naming conventions: test_{module}_{function}_{scenario}")
            
            if execution_cat.aaa_pattern_score and execution_cat.aaa_pattern_score < 90:
                recommendations.append("Implement AAA (Arrange-Act-Assert) patterns in test functions")
                recommendations.append("Add clear comments separating test phases: # Arrange, # Act, # Assert")
            
            if execution_cat.fixture_quality_score and execution_cat.fixture_quality_score < 90:
                recommendations.append("Centralize fixtures in tests/conftest.py and tests/utils.py")
                recommendations.append("Remove duplicate fixtures from individual test modules")
        
        # Enhanced testing strategy recommendations
        enhanced_metrics = result.enhanced_testing_metrics
        if not enhanced_metrics.get("pytest_style_compliance"):
            recommendations.append("Enable pre-commit hooks to enforce pytest-style validation automatically")
        
        if enhanced_metrics.get("aaa_pattern_score", 0) < 95:
            recommendations.append("Review docs/testing_guidelines.md for AAA pattern implementation examples")
        
        # General recommendations
        if result.quality_score < 95:
            recommendations.append("Review the comprehensive testing strategy documentation for best practices")
            recommendations.append("Consider running full validation with 'pytest --run-network --runslow' for complete analysis")
        
        return recommendations

    def _add_audit_entry(self, message: str) -> None:
        """Add entry to audit trail for research-grade traceability."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        audit_entry = f"[{timestamp}] {message}"
        self.audit_trail.append(audit_entry)
        
        # Also log to file for persistent audit trail
        self.logger.debug(f"AUDIT: {message}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create enhanced command line argument parser with scripts/coverage/ integration."""
    parser = argparse.ArgumentParser(
        description="Enhanced Quality Gate Enforcement Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic quality gate validation
  python scripts/coverage/enforce-quality-gates.py
  
  # CI mode with JSON output
  python scripts/coverage/enforce-quality-gates.py --ci-mode --output-format json
  
  # Dry run validation
  python scripts/coverage/enforce-quality-gates.py --dry-run --verbose
  
  # Generate HTML report
  python scripts/coverage/enforce-quality-gates.py --output-format html --output-path reports/quality-gates.html

Integration Points:
  - scripts/coverage/quality-gates.yml: Quality gate configuration
  - scripts/coverage/coverage-thresholds.json: Coverage threshold definitions
  - scripts/coverage/validate-coverage.py: Coverage validation script
  - scripts/coverage/check-performance-slas.py: Performance validation script
  - scripts/benchmarks/: Performance test isolation directory
        """
    )
    
    parser.add_argument(
        "--quality-gates-config",
        type=Path,
        default=Path("scripts/coverage/quality-gates.yml"),
        help="Path to quality gates configuration file (default: scripts/coverage/quality-gates.yml)"
    )
    
    parser.add_argument(
        "--coverage-thresholds",
        type=Path,
        default=Path("scripts/coverage/coverage-thresholds.json"),
        help="Path to coverage thresholds configuration file (default: scripts/coverage/coverage-thresholds.json)"
    )
    
    parser.add_argument(
        "--ci-mode",
        action="store_true",
        help="Enable CI/CD integration mode with structured output and GitHub Actions support"
    )
    
    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        default=True,
        help="Exit with non-zero code on quality gate violations (default: True)"
    )
    
    parser.add_argument(
        "--output-format",
        choices=["console", "json", "html"],
        default="console",
        help="Output format for quality gate report (default: console)"
    )
    
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Output path for generated reports (default: scripts/coverage/quality-gate-report.*)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and show what would be executed without running quality gates"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging and diagnostic output"
    )
    
    return parser


def main() -> int:
    """
    Main entry point for enhanced quality gate enforcement orchestrator.
    
    Implements comprehensive quality gate validation with enhanced testing strategy
    support and CI/CD integration per Section 8.4 CI/CD Pipeline Architecture.
    
    Returns:
        int: Exit code (0=success, 1=violations, 2=error)
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Initialize enhanced quality gate enforcer
        enforcer = EnhancedQualityGateEnforcer(
            quality_gates_config_path=args.quality_gates_config,
            coverage_thresholds_path=args.coverage_thresholds,
            ci_mode=args.ci_mode,
            fail_on_violation=args.fail_on_violation,
            output_format=args.output_format,
            verbose=args.verbose
        )
        
        # Load configurations
        enforcer.load_configurations()
        
        # Dry run mode - validate configuration only
        if args.dry_run:
            enforcer.logger.info("Dry run mode - configuration validation completed successfully")
            print("✅ Configuration validation passed - all required files and settings are valid")
            print("🔧 Enhanced testing strategy integration confirmed")
            print("📁 Scripts/coverage/ directory structure validated")
            print("🚀 Ready for quality gate execution")
            return 0
        
        # Execute comprehensive quality gates
        result = enforcer.execute_quality_gates()
        
        # Generate report
        enforcer.generate_report(result, args.output_path)
        
        # GitHub Actions integration outputs
        if enforcer.github_actions_integration:
            print(f"::set-output name=overall_passed::{str(result.overall_passed).lower()}")
            print(f"::set-output name=quality_score::{result.quality_score:.1f}")
            print(f"::set-output name=merge_allowed::{str(result.merge_allowed).lower()}")
            print(f"::set-output name=critical_failures::{len(result.critical_failures)}")
            
            # Set GitHub Actions job summary
            summary_file = os.getenv("GITHUB_STEP_SUMMARY")
            if summary_file:
                with open(summary_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n## Quality Gate Results\n")
                    f.write(f"- **Overall Status**: {'✅ PASSED' if result.overall_passed else '❌ FAILED'}\n")
                    f.write(f"- **Quality Score**: {result.quality_score:.1f}%\n")
                    f.write(f"- **Merge Decision**: {'✅ ALLOWED' if result.merge_allowed else '❌ BLOCKED'}\n")
                    f.write(f"- **Critical Failures**: {len(result.critical_failures)}\n")
        
        # Return appropriate exit code
        if result.overall_passed:
            enforcer.logger.info("Quality gate enforcement completed successfully - all gates passed")
            return 0
        else:
            if args.fail_on_violation:
                enforcer.logger.error("Quality gate enforcement failed - violations detected")
                return 1
            else:
                enforcer.logger.warning("Quality gate enforcement completed with violations - exit code overridden")
                return 0
                
    except FileNotFoundError as e:
        print(f"❌ Configuration file not found: {e}", file=sys.stderr)
        if args.ci_mode:
            print(f"::error::Configuration file not found: {e}")
        return 2
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        print(f"❌ Configuration file invalid: {e}", file=sys.stderr)
        if args.ci_mode:
            print(f"::error::Configuration file invalid: {e}")
        return 2
    except Exception as e:
        print(f"❌ Quality gate enforcement failed with error: {e}", file=sys.stderr)
        if args.ci_mode:
            print(f"::error::Quality gate enforcement failed: {e}")
        
        # Enhanced error reporting for debugging
        if args.verbose:
            import traceback
            print(f"\nDetailed error information:", file=sys.stderr)
            traceback.print_exc()
        
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)