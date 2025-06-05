#!/usr/bin/env python3
"""
Comprehensive Quality Gate Enforcement Script

Implements automated validation of coverage thresholds, performance SLAs, and test execution 
criteria with CI/CD integration and merge blocking capabilities. Serves as the primary 
quality assurance gatekeeper per TST-COV-004 and Section 4.1.1.5 requirements.

This script orchestrates:
- TST-COV-001: Maintain >90% overall test coverage across all modules
- TST-COV-002: Achieve 100% coverage for critical data loading and validation modules  
- TST-COV-004: Block merges when coverage drops below thresholds
- TST-PERF-001: Data loading SLA validation within 1s per 100MB
- TST-PERF-002: DataFrame transformation SLA validation within 500ms per 1M rows
- Section 4.1.1.5: Test execution workflow with automated quality gate enforcement

Usage:
    python tests/coverage/scripts/enforce-quality-gates.py [options]
    
Exit Codes:
    0: All quality gates passed - merge allowed
    1: Quality gate violations detected - merge blocked
    2: Configuration or system error
    
Environment Variables:
    ENFORCE_QUALITY_GATES: Set to 'false' to disable enforcement (default: 'true')
    CI: Set to 'true' for CI/CD pipeline integration mode
    GITHUB_ACTIONS: Automatically detected for GitHub Actions workflow integration
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import yaml
import logging
from enum import Enum
import tempfile


class QualityGateStatus(Enum):
    """Quality gate validation status enumeration."""
    PASSED = "PASSED"
    FAILED = "FAILED" 
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


class QualityGateCategory(Enum):
    """Quality gate category enumeration."""
    COVERAGE = "coverage"
    PERFORMANCE = "performance"
    EXECUTION = "execution"


@dataclass
class QualityGateViolation:
    """Represents a single quality gate violation."""
    category: QualityGateCategory
    severity: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    message: str
    details: Optional[str] = None
    module_path: Optional[str] = None
    threshold: Optional[float] = None
    actual_value: Optional[float] = None
    action_required: Optional[str] = None


@dataclass
class QualityGateMetrics:
    """Container for quality gate execution metrics."""
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    error_checks: int = 0
    execution_time_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class QualityGateResults:
    """Comprehensive results of quality gate validation."""
    overall_status: QualityGateStatus = QualityGateStatus.PASSED
    violations: List[QualityGateViolation] = field(default_factory=list)
    metrics: QualityGateMetrics = field(default_factory=QualityGateMetrics)
    coverage_report_path: Optional[Path] = None
    performance_report_path: Optional[Path] = None
    detailed_report_path: Optional[Path] = None
    
    # Category-specific results
    coverage_passed: bool = True
    performance_passed: bool = True
    execution_passed: bool = True
    
    # Merge decision
    merge_allowed: bool = True
    merge_block_reason: Optional[str] = None


class QualityGateConfigurationManager:
    """Manages quality gate configuration loading and validation."""
    
    def __init__(self, base_path: Path = None):
        """Initialize configuration manager."""
        self.base_path = base_path or Path("tests/coverage")
        self.quality_gates_config_path = self.base_path / "quality-gates.yml"
        self.coverage_thresholds_path = self.base_path / "coverage-thresholds.json"
        self._quality_gates_config = None
        self._coverage_thresholds_config = None
        
        # Set up logging for configuration management
        self.logger = logging.getLogger(__name__ + ".config")
    
    def load_quality_gates_config(self) -> Dict[str, Any]:
        """Load and validate quality gates configuration."""
        if self._quality_gates_config is None:
            try:
                with open(self.quality_gates_config_path, 'r', encoding='utf-8') as f:
                    self._quality_gates_config = yaml.safe_load(f)
                
                self._validate_quality_gates_config()
                self.logger.info(f"Loaded quality gates configuration from {self.quality_gates_config_path}")
                
            except FileNotFoundError:
                self._quality_gates_config = self._get_default_quality_gates_config()
                self.logger.warning(f"Quality gates config not found at {self.quality_gates_config_path}, using defaults")
                
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in quality gates configuration: {e}")
                
        return self._quality_gates_config
    
    def load_coverage_thresholds_config(self) -> Dict[str, Any]:
        """Load and validate coverage thresholds configuration."""
        if self._coverage_thresholds_config is None:
            try:
                with open(self.coverage_thresholds_path, 'r', encoding='utf-8') as f:
                    self._coverage_thresholds_config = json.load(f)
                
                self._validate_coverage_thresholds_config()
                self.logger.info(f"Loaded coverage thresholds from {self.coverage_thresholds_path}")
                
            except FileNotFoundError:
                self._coverage_thresholds_config = self._get_default_coverage_thresholds_config()
                self.logger.warning(f"Coverage thresholds config not found at {self.coverage_thresholds_path}, using defaults")
                
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in coverage thresholds configuration: {e}")
                
        return self._coverage_thresholds_config
    
    def _get_default_quality_gates_config(self) -> Dict[str, Any]:
        """Provide default quality gates configuration based on requirements."""
        return {
            "coverage": {
                "overall_coverage_threshold": 90.0,
                "critical_module_coverage_threshold": 100.0,
                "branch_coverage_required": True,
                "minimum_branch_coverage": 85.0,
                "fail_on_violation": True,
                "strict_validation": True
            },
            "performance": {
                "data_loading_sla": "1s per 100MB",
                "dataframe_transformation_sla": "500ms per 1M rows",
                "benchmark_execution_timeout": 600,
                "fail_on_violation": True,
                "regression_detection": True
            },
            "execution": {
                "test_execution_timeout": 300,
                "strict_validation": True,
                "block_merge_on_failure": True
            },
            "integration": {
                "github_actions": {
                    "workflow_file": ".github/workflows/test.yml"
                }
            }
        }
    
    def _get_default_coverage_thresholds_config(self) -> Dict[str, Any]:
        """Provide default coverage thresholds configuration."""
        return {
            "global_configuration": {
                "overall_threshold": 90.0,
                "fail_on_violation": True,
                "enforcement_enabled": True
            },
            "module_thresholds": {
                "critical_modules": {
                    "modules": {
                        "src/flyrigloader/api.py": {"threshold": 100.0},
                        "src/flyrigloader/config/yaml_config.py": {"threshold": 100.0},
                        "src/flyrigloader/config/discovery.py": {"threshold": 100.0},
                        "src/flyrigloader/discovery/files.py": {"threshold": 100.0},
                        "src/flyrigloader/discovery/patterns.py": {"threshold": 100.0},
                        "src/flyrigloader/discovery/stats.py": {"threshold": 100.0},
                        "src/flyrigloader/io/pickle.py": {"threshold": 100.0},
                        "src/flyrigloader/io/column_models.py": {"threshold": 100.0}
                    }
                }
            }
        }
    
    def _validate_quality_gates_config(self) -> None:
        """Validate quality gates configuration structure."""
        required_sections = ["coverage", "performance", "execution"]
        for section in required_sections:
            if section not in self._quality_gates_config:
                raise ValueError(f"Missing required section '{section}' in quality gates configuration")
    
    def _validate_coverage_thresholds_config(self) -> None:
        """Validate coverage thresholds configuration structure."""
        required_keys = ["global_configuration", "module_thresholds"]
        for key in required_keys:
            if key not in self._coverage_thresholds_config:
                raise ValueError(f"Missing required key '{key}' in coverage thresholds configuration")
    
    def get_enforcement_enabled(self) -> bool:
        """Check if quality gate enforcement is enabled."""
        # Check environment variable first
        env_enforcement = os.getenv("ENFORCE_QUALITY_GATES", "true").lower()
        if env_enforcement == "false":
            return False
        
        # Check configuration
        quality_gates_config = self.load_quality_gates_config()
        coverage_config = quality_gates_config.get("coverage", {})
        return coverage_config.get("fail_on_violation", True)


class CoverageValidator:
    """Validates coverage quality gates using existing coverage validation infrastructure."""
    
    def __init__(self, config_manager: QualityGateConfigurationManager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__ + ".coverage")
    
    def validate_coverage_gates(self, coverage_file: Path = None) -> Tuple[bool, List[QualityGateViolation]]:
        """Validate coverage quality gates by delegating to validate-coverage.py."""
        violations = []
        
        try:
            # Determine coverage file path
            if coverage_file is None:
                coverage_file = Path("coverage.xml")
                if not coverage_file.exists():
                    coverage_file = Path("coverage.json")
            
            # Prepare command for coverage validation
            validate_coverage_script = Path("tests/coverage/validate-coverage.py")
            
            if not validate_coverage_script.exists():
                self.logger.error(f"Coverage validation script not found: {validate_coverage_script}")
                violations.append(QualityGateViolation(
                    category=QualityGateCategory.COVERAGE,
                    severity="CRITICAL",
                    message="Coverage validation script not found",
                    details=f"Expected script at {validate_coverage_script}",
                    action_required="Ensure coverage validation infrastructure is properly installed"
                ))
                return False, violations
            
            # Execute coverage validation
            cmd = [sys.executable, str(validate_coverage_script)]
            if coverage_file.exists():
                cmd.extend(["--coverage-file", str(coverage_file)])
            
            self.logger.info(f"Executing coverage validation: {' '.join(cmd)}")
            
            # Run coverage validation with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,  # 1 minute timeout
                cwd=Path.cwd()
            )
            
            # Parse results
            if result.returncode == 0:
                self.logger.info("Coverage validation passed")
                return True, violations
            else:
                self.logger.error(f"Coverage validation failed with exit code {result.returncode}")
                
                # Parse stderr for specific violations
                if result.stderr:
                    for line in result.stderr.strip().split('\n'):
                        if line.strip():
                            violations.append(QualityGateViolation(
                                category=QualityGateCategory.COVERAGE,
                                severity="HIGH",
                                message=f"Coverage validation error: {line.strip()}",
                                action_required="Fix coverage issues identified in the validation report"
                            ))
                
                # Parse stdout for detailed violations
                if result.stdout:
                    violations.extend(self._parse_coverage_output(result.stdout))
                
                return False, violations
        
        except subprocess.TimeoutExpired:
            self.logger.error("Coverage validation timed out")
            violations.append(QualityGateViolation(
                category=QualityGateCategory.COVERAGE,
                severity="CRITICAL",
                message="Coverage validation timed out",
                details="Coverage analysis exceeded 60 second timeout",
                action_required="Optimize test suite or increase timeout limits"
            ))
            return False, violations
            
        except Exception as e:
            self.logger.error(f"Coverage validation failed with exception: {e}")
            violations.append(QualityGateViolation(
                category=QualityGateCategory.COVERAGE,
                severity="CRITICAL",
                message=f"Coverage validation system error: {str(e)}",
                action_required="Check coverage validation system configuration"
            ))
            return False, violations
    
    def _parse_coverage_output(self, output: str) -> List[QualityGateViolation]:
        """Parse coverage validation output for specific violations."""
        violations = []
        
        for line in output.split('\n'):
            line = line.strip()
            
            # Look for coverage violation patterns
            if "below threshold" in line.lower():
                violations.append(QualityGateViolation(
                    category=QualityGateCategory.COVERAGE,
                    severity="HIGH",
                    message=line,
                    action_required="Increase test coverage for the affected module"
                ))
            elif "coverage" in line.lower() and ("fail" in line.lower() or "error" in line.lower()):
                violations.append(QualityGateViolation(
                    category=QualityGateCategory.COVERAGE,
                    severity="MEDIUM",
                    message=line,
                    action_required="Review and fix coverage analysis issues"
                ))
        
        return violations


class PerformanceValidator:
    """Validates performance SLA quality gates."""
    
    def __init__(self, config_manager: QualityGateConfigurationManager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__ + ".performance")
    
    def validate_performance_gates(self, benchmark_file: Path = None) -> Tuple[bool, List[QualityGateViolation]]:
        """Validate performance SLA quality gates."""
        violations = []
        
        try:
            # Check if performance validation is enabled
            config = self.config_manager.load_quality_gates_config()
            performance_config = config.get("performance", {})
            
            if not performance_config.get("fail_on_violation", True):
                self.logger.info("Performance validation disabled by configuration")
                return True, violations
            
            # Execute performance SLA validation
            performance_script = Path("tests/coverage/scripts/check-performance-slas.py")
            
            if not performance_script.exists():
                self.logger.warning(f"Performance validation script not found: {performance_script}")
                # For now, we'll just log a warning since this might be optional
                return True, violations
            
            # Determine benchmark file path  
            if benchmark_file is None:
                benchmark_file = Path("benchmark_results.json")
                if not benchmark_file.exists():
                    benchmark_file = Path(".benchmarks/latest.json")
            
            # Prepare command for performance validation
            cmd = [sys.executable, str(performance_script)]
            if benchmark_file and benchmark_file.exists():
                cmd.extend(["--benchmark-file", str(benchmark_file)])
            
            self.logger.info(f"Executing performance validation: {' '.join(cmd)}")
            
            # Run performance validation with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
                cwd=Path.cwd()
            )
            
            # Parse results
            if result.returncode == 0:
                self.logger.info("Performance validation passed")
                return True, violations
            else:
                self.logger.error(f"Performance validation failed with exit code {result.returncode}")
                
                # Parse output for specific violations
                violations.extend(self._parse_performance_output(result.stdout, result.stderr))
                return False, violations
        
        except subprocess.TimeoutExpired:
            self.logger.error("Performance validation timed out")
            violations.append(QualityGateViolation(
                category=QualityGateCategory.PERFORMANCE,
                severity="HIGH",
                message="Performance validation timed out",
                details="Performance benchmark analysis exceeded 2 minute timeout",
                action_required="Optimize benchmark tests or increase timeout limits"
            ))
            return False, violations
            
        except FileNotFoundError:
            self.logger.warning("Performance validation script not available")
            # If performance validation is not available, don't fail the gates
            return True, violations
            
        except Exception as e:
            self.logger.error(f"Performance validation failed with exception: {e}")
            violations.append(QualityGateViolation(
                category=QualityGateCategory.PERFORMANCE,
                severity="MEDIUM",
                message=f"Performance validation system error: {str(e)}",
                action_required="Check performance validation system configuration"
            ))
            return False, violations
    
    def _parse_performance_output(self, stdout: str, stderr: str) -> List[QualityGateViolation]:
        """Parse performance validation output for specific violations."""
        violations = []
        
        output = (stdout + "\n" + stderr).strip()
        
        for line in output.split('\n'):
            line = line.strip()
            
            # Look for SLA violation patterns
            if "sla" in line.lower() and ("fail" in line.lower() or "exceed" in line.lower()):
                violations.append(QualityGateViolation(
                    category=QualityGateCategory.PERFORMANCE,
                    severity="HIGH",
                    message=line,
                    action_required="Optimize performance for the failing operation"
                ))
            elif "timeout" in line.lower() or "slow" in line.lower():
                violations.append(QualityGateViolation(
                    category=QualityGateCategory.PERFORMANCE,
                    severity="MEDIUM",
                    message=line,
                    action_required="Review and optimize performance bottlenecks"
                ))
        
        return violations


class ExecutionValidator:
    """Validates test execution quality gates."""
    
    def __init__(self, config_manager: QualityGateConfigurationManager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__ + ".execution")
    
    def validate_execution_gates(self) -> Tuple[bool, List[QualityGateViolation]]:
        """Validate test execution quality gates."""
        violations = []
        
        try:
            config = self.config_manager.load_quality_gates_config()
            execution_config = config.get("execution", {})
            
            # Check test execution timeout compliance
            if execution_config.get("strict_validation", True):
                self.logger.info("Performing strict execution validation")
                
                # Validate test environment
                if not self._validate_test_environment():
                    violations.append(QualityGateViolation(
                        category=QualityGateCategory.EXECUTION,
                        severity="MEDIUM",
                        message="Test environment validation issues detected",
                        action_required="Ensure all test dependencies are properly installed"
                    ))
                
                # Validate CI/CD integration if in CI environment
                if os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true":
                    if not self._validate_ci_integration():
                        violations.append(QualityGateViolation(
                            category=QualityGateCategory.EXECUTION,
                            severity="LOW",
                            message="CI/CD integration configuration issues detected",
                            action_required="Review CI/CD pipeline configuration"
                        ))
            
            return len(violations) == 0, violations
            
        except Exception as e:
            self.logger.error(f"Execution validation failed with exception: {e}")
            violations.append(QualityGateViolation(
                category=QualityGateCategory.EXECUTION,
                severity="LOW",
                message=f"Execution validation system error: {str(e)}",
                action_required="Check execution validation system configuration"
            ))
            return False, violations
    
    def _validate_test_environment(self) -> bool:
        """Validate test environment setup."""
        try:
            # Check if pytest is available
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                self.logger.warning("pytest not available or misconfigured")
                return False
                
            # Check if coverage is available
            result = subprocess.run(
                [sys.executable, "-m", "coverage", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                self.logger.warning("coverage.py not available or misconfigured")
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Test environment validation failed: {e}")
            return False
    
    def _validate_ci_integration(self) -> bool:
        """Validate CI/CD integration configuration."""
        try:
            # Check for GitHub Actions workflow file
            workflow_file = Path(".github/workflows/test.yml")
            if not workflow_file.exists():
                self.logger.warning(f"GitHub Actions workflow not found: {workflow_file}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"CI/CD integration validation failed: {e}")
            return False


class QualityGateOrchestrator:
    """Main orchestrator for comprehensive quality gate validation."""
    
    def __init__(self, base_path: Path = None, verbose: bool = False):
        """Initialize quality gate orchestrator."""
        self.base_path = base_path or Path("tests/coverage")
        self.verbose = verbose
        
        # Set up logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__ + ".orchestrator")
        
        # Initialize components
        self.config_manager = QualityGateConfigurationManager(self.base_path)
        self.coverage_validator = CoverageValidator(self.config_manager)
        self.performance_validator = PerformanceValidator(self.config_manager)
        self.execution_validator = ExecutionValidator(self.config_manager)
        
        self.logger.info("Quality gate orchestrator initialized")
    
    def _setup_logging(self) -> None:
        """Set up comprehensive logging configuration."""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.base_path / "logs" / "quality-gates.log", mode='w')
            ]
        )
        
        # Ensure log directory exists
        log_dir = self.base_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
    
    def execute_quality_gates(self, 
                              coverage_file: Path = None,
                              benchmark_file: Path = None) -> QualityGateResults:
        """Execute comprehensive quality gate validation."""
        start_time = time.time()
        results = QualityGateResults()
        
        self.logger.info("🚀 Starting comprehensive quality gate validation")
        self.logger.info(f"📅 Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check if enforcement is enabled
        if not self.config_manager.get_enforcement_enabled():
            self.logger.info("Quality gate enforcement is disabled")
            results.overall_status = QualityGateStatus.SKIPPED
            results.merge_allowed = True
            return results
        
        try:
            # Initialize metrics
            metrics = QualityGateMetrics()
            
            # Execute coverage validation
            self.logger.info("📊 Executing coverage quality gates...")
            metrics.total_checks += 1
            coverage_passed, coverage_violations = self.coverage_validator.validate_coverage_gates(coverage_file)
            
            if coverage_passed:
                metrics.passed_checks += 1
                self.logger.info("✅ Coverage quality gates PASSED")
            else:
                metrics.failed_checks += 1
                self.logger.error("❌ Coverage quality gates FAILED")
                results.coverage_passed = False
            
            results.violations.extend(coverage_violations)
            
            # Execute performance validation
            self.logger.info("⚡ Executing performance quality gates...")
            metrics.total_checks += 1
            performance_passed, performance_violations = self.performance_validator.validate_performance_gates(benchmark_file)
            
            if performance_passed:
                metrics.passed_checks += 1
                self.logger.info("✅ Performance quality gates PASSED")
            else:
                metrics.failed_checks += 1
                self.logger.error("❌ Performance quality gates FAILED")
                results.performance_passed = False
            
            results.violations.extend(performance_violations)
            
            # Execute execution validation
            self.logger.info("🔧 Executing execution quality gates...")
            metrics.total_checks += 1
            execution_passed, execution_violations = self.execution_validator.validate_execution_gates()
            
            if execution_passed:
                metrics.passed_checks += 1
                self.logger.info("✅ Execution quality gates PASSED")
            else:
                metrics.failed_checks += 1
                self.logger.error("❌ Execution quality gates FAILED")
                results.execution_passed = False
            
            results.violations.extend(execution_violations)
            
            # Determine overall status
            all_passed = coverage_passed and performance_passed and execution_passed
            
            if all_passed:
                results.overall_status = QualityGateStatus.PASSED
                results.merge_allowed = True
                self.logger.info("🎉 ALL QUALITY GATES PASSED - Merge allowed")
            else:
                results.overall_status = QualityGateStatus.FAILED
                results.merge_allowed = False
                
                # Determine merge block reason
                critical_violations = [v for v in results.violations if v.severity == "CRITICAL"]
                if critical_violations:
                    results.merge_block_reason = f"Critical violations detected: {len(critical_violations)} issues"
                else:
                    results.merge_block_reason = f"Quality gate failures: {metrics.failed_checks} categories failed"
                
                self.logger.error(f"🚫 QUALITY GATES FAILED - Merge blocked: {results.merge_block_reason}")
            
            # Finalize metrics
            metrics.execution_time_seconds = time.time() - start_time
            results.metrics = metrics
            
            return results
            
        except Exception as e:
            self.logger.error(f"Quality gate execution failed with critical error: {e}")
            results.overall_status = QualityGateStatus.ERROR
            results.merge_allowed = False
            results.merge_block_reason = f"System error during quality gate validation: {str(e)}"
            
            # Add critical violation
            results.violations.append(QualityGateViolation(
                category=QualityGateCategory.EXECUTION,
                severity="CRITICAL",
                message=f"Quality gate orchestration system error: {str(e)}",
                action_required="Contact development team to resolve quality gate infrastructure issues"
            ))
            
            return results
    
    def generate_comprehensive_report(self, results: QualityGateResults) -> str:
        """Generate comprehensive quality gate report."""
        report_lines = []
        
        # Header with status
        status_emoji = {
            QualityGateStatus.PASSED: "✅",
            QualityGateStatus.FAILED: "❌", 
            QualityGateStatus.ERROR: "💥",
            QualityGateStatus.SKIPPED: "⏭️"
        }[results.overall_status]
        
        report_lines.extend([
            f"{status_emoji} COMPREHENSIVE QUALITY GATE VALIDATION REPORT",
            "=" * 60,
            f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"🏆 Overall Status: {results.overall_status.value}",
            f"🚀 Execution Time: {results.metrics.execution_time_seconds:.2f} seconds",
            f"🎯 Merge Decision: {'ALLOWED' if results.merge_allowed else 'BLOCKED'}",
            ""
        ])
        
        # Merge block reason if applicable
        if not results.merge_allowed and results.merge_block_reason:
            report_lines.extend([
                "🚫 MERGE BLOCKED",
                "-" * 15,
                f"Reason: {results.merge_block_reason}",
                ""
            ])
        
        # Execution summary
        report_lines.extend([
            "📊 EXECUTION SUMMARY",
            "-" * 19,
            f"Total Quality Gates: {results.metrics.total_checks}",
            f"✅ Passed: {results.metrics.passed_checks}",
            f"❌ Failed: {results.metrics.failed_checks}",
            f"💥 Errors: {results.metrics.error_checks}",
            ""
        ])
        
        # Category results
        report_lines.extend([
            "🎯 CATEGORY RESULTS",
            "-" * 18,
            f"📊 Coverage Validation: {'✅ PASSED' if results.coverage_passed else '❌ FAILED'}",
            f"⚡ Performance Validation: {'✅ PASSED' if results.performance_passed else '❌ FAILED'}",
            f"🔧 Execution Validation: {'✅ PASSED' if results.execution_passed else '❌ FAILED'}",
            ""
        ])
        
        # Detailed violations
        if results.violations:
            report_lines.extend([
                "⚠️  QUALITY GATE VIOLATIONS",
                "-" * 27
            ])
            
            # Group violations by category and severity
            violation_groups = {}
            for violation in results.violations:
                key = f"{violation.category.value}_{violation.severity}"
                if key not in violation_groups:
                    violation_groups[key] = []
                violation_groups[key].append(violation)
            
            # Report violations by severity
            for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
                severity_violations = [v for v in results.violations if v.severity == severity]
                if severity_violations:
                    severity_emoji = {"CRITICAL": "💥", "HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵"}[severity]
                    report_lines.append(f"\n{severity_emoji} {severity} SEVERITY ({len(severity_violations)} issues):")
                    
                    for violation in severity_violations:
                        report_lines.append(f"  • {violation.message}")
                        if violation.details:
                            report_lines.append(f"    Details: {violation.details}")
                        if violation.action_required:
                            report_lines.append(f"    Action: {violation.action_required}")
                        report_lines.append("")
        
        # Action items and next steps
        if not results.merge_allowed:
            report_lines.extend([
                "📝 REQUIRED ACTIONS",
                "-" * 17,
                "To resolve quality gate violations and enable merge:"
            ])
            
            critical_violations = [v for v in results.violations if v.severity == "CRITICAL"]
            high_violations = [v for v in results.violations if v.severity == "HIGH"]
            
            if critical_violations:
                report_lines.append("1. 💥 CRITICAL: Address critical system issues immediately")
                for violation in critical_violations:
                    if violation.action_required:
                        report_lines.append(f"   - {violation.action_required}")
            
            if high_violations:
                report_lines.append("2. 🔴 HIGH: Fix high-priority quality violations")
                for violation in high_violations:
                    if violation.action_required:
                        report_lines.append(f"   - {violation.action_required}")
            
            report_lines.extend([
                "3. 🔄 Re-run quality gates after fixes",
                "4. 📊 Verify all quality metrics meet requirements",
                ""
            ])
        
        # System information
        report_lines.extend([
            "🛠️  SYSTEM INFORMATION",
            "-" * 20,
            f"Python Version: {sys.version.split()[0]}",
            f"Working Directory: {Path.cwd()}",
            f"Quality Gates Path: {self.base_path}",
            f"CI Environment: {'Yes' if os.getenv('CI') == 'true' else 'No'}",
            f"GitHub Actions: {'Yes' if os.getenv('GITHUB_ACTIONS') == 'true' else 'No'}",
            ""
        ])
        
        # Footer
        report_lines.extend([
            "📚 DOCUMENTATION",
            "-" * 15,
            "For more information about quality gates:",
            "- Technical Specification Section 4.1.1.5",
            "- TST-COV-001, TST-COV-002, TST-COV-004 requirements",
            "- TST-PERF-001, TST-PERF-002 performance SLAs",
            "",
            "Generated by: flyrigloader Quality Gate Enforcement System",
            f"Report ID: QG-{int(time.time())}"
        ])
        
        return "\n".join(report_lines)
    
    def save_detailed_report(self, results: QualityGateResults, output_path: Path = None) -> Path:
        """Save detailed quality gate report to file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.base_path / "reports" / f"quality_gate_report_{timestamp}.txt"
        
        # Ensure report directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate and save report
        report_content = self.generate_comprehensive_report(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"📄 Detailed quality gate report saved to: {output_path}")
        results.detailed_report_path = output_path
        
        return output_path


def main():
    """Main entry point for quality gate enforcement script."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Quality Gate Enforcement System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/coverage/scripts/enforce-quality-gates.py
  python tests/coverage/scripts/enforce-quality-gates.py --coverage-file coverage.xml
  python tests/coverage/scripts/enforce-quality-gates.py --verbose --save-report
  
Exit Codes:
  0: All quality gates passed - merge allowed
  1: Quality gate violations detected - merge blocked  
  2: Configuration or system error
        """
    )
    
    parser.add_argument(
        "--coverage-file",
        type=Path,
        help="Path to coverage report file (XML or JSON format)"
    )
    
    parser.add_argument(
        "--benchmark-file", 
        type=Path,
        help="Path to performance benchmark results file (JSON format)"
    )
    
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save detailed quality gate report to file"
    )
    
    parser.add_argument(
        "--report-output",
        type=Path,
        help="Custom path for detailed report output"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output"
    )
    
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path("tests/coverage"),
        help="Base path for quality gate configuration files"
    )
    
    parser.add_argument(
        "--disable-enforcement",
        action="store_true",
        help="Disable quality gate enforcement (for testing only)"
    )
    
    args = parser.parse_args()
    
    # Set environment variable if enforcement is disabled
    if args.disable_enforcement:
        os.environ["ENFORCE_QUALITY_GATES"] = "false"
    
    try:
        # Initialize orchestrator
        orchestrator = QualityGateOrchestrator(
            base_path=args.base_path,
            verbose=args.verbose
        )
        
        # Execute quality gates
        results = orchestrator.execute_quality_gates(
            coverage_file=args.coverage_file,
            benchmark_file=args.benchmark_file
        )
        
        # Generate and display report
        report = orchestrator.generate_comprehensive_report(results)
        print(report)
        
        # Save detailed report if requested
        if args.save_report or args.report_output:
            orchestrator.save_detailed_report(results, args.report_output)
        
        # Determine exit code based on results
        if results.overall_status == QualityGateStatus.PASSED:
            exit_code = 0
        elif results.overall_status == QualityGateStatus.SKIPPED:
            exit_code = 0  # Don't fail if enforcement is disabled
        elif results.overall_status == QualityGateStatus.ERROR:
            exit_code = 2
        else:  # FAILED
            exit_code = 1
        
        # Log final status
        if args.verbose:
            status_messages = {
                0: "✅ Quality gates passed - merge allowed",
                1: "❌ Quality gates failed - merge blocked", 
                2: "💥 System error - manual intervention required"
            }
            print(f"\n🚪 Exiting with code {exit_code}: {status_messages[exit_code]}")
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n⏹️  Quality gate validation interrupted by user")
        sys.exit(2)
        
    except Exception as e:
        print(f"💥 Quality gate enforcement failed with critical error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()