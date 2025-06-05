#!/usr/bin/env python3
"""
Automated Coverage Validation Script

Implements comprehensive quality gate enforcement with module-specific threshold validation,
performance SLA checking, and CI/CD integration support. Provides programmatic coverage 
analysis enabling automated merge blocking per TST-COV-004 requirements.

This script validates:
- TST-COV-001: Maintain >90% overall test coverage across all modules
- TST-COV-002: Achieve 100% coverage for critical data loading and validation modules
- TST-COV-004: Block merges when coverage drops below thresholds
- TST-PERF-001: Data loading SLA validation within 1s per 100MB
- Section 4.1.1.5: Test execution workflow with automated quality gate enforcement

Usage:
    python tests/coverage/validate-coverage.py [--coverage-file coverage.xml] [--benchmark-file benchmark.json]
    
Exit Codes:
    0: All quality gates passed
    1: Quality gate violations detected (blocks merge)
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import yaml
import re
from datetime import datetime


@dataclass
class CoverageMetrics:
    """Container for coverage metrics data."""
    line_coverage: float
    branch_coverage: float
    total_lines: int
    covered_lines: int
    total_branches: int
    covered_branches: int
    missing_lines: List[int] = field(default_factory=list)


@dataclass
class ModuleCoverageReport:
    """Coverage report for a specific module."""
    module_name: str
    metrics: CoverageMetrics
    threshold: float
    is_critical: bool
    violations: List[str] = field(default_factory=list)


@dataclass
class PerformanceMetrics:
    """Container for performance benchmark metrics."""
    test_name: str
    mean_time: float
    std_dev: float
    min_time: float
    max_time: float
    iterations: int


@dataclass
class QualityGateResults:
    """Results of quality gate validation."""
    passed: bool = True
    coverage_violations: List[str] = field(default_factory=list)
    performance_violations: List[str] = field(default_factory=list)
    module_reports: List[ModuleCoverageReport] = field(default_factory=list)
    overall_coverage: float = 0.0
    overall_branch_coverage: float = 0.0


class CoverageThresholdLoader:
    """Loads and validates coverage threshold configuration."""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._config = None
    
    def load_config(self) -> Dict[str, Any]:
        """Load coverage threshold configuration from JSON file."""
        if self._config is None:
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
                self._validate_config()
            except FileNotFoundError:
                # Provide default configuration if file doesn't exist
                self._config = self._get_default_config()
                print(f"Warning: Coverage thresholds file not found at {self.config_path}, using defaults")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in coverage thresholds file: {e}")
        
        return self._config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Provide default coverage configuration based on requirements."""
        return {
            "overall_threshold": 90.0,
            "branch_coverage_required": True,
            "critical_modules": {
                "src/flyrigloader/api.py": 100.0,
                "src/flyrigloader/config/": 100.0,
                "src/flyrigloader/discovery/": 100.0,
                "src/flyrigloader/io/": 100.0
            },
            "module_thresholds": {
                "src/flyrigloader/utils/": 95.0,
                "src/flyrigloader/__init__.py": 90.0
            },
            "exclude_patterns": [
                "*/test_*",
                "*/conftest.py",
                "*/__pycache__/*"
            ],
            "quality_gates": {
                "fail_on_violation": True,
                "require_branch_coverage": True,
                "minimum_branch_coverage": 85.0
            }
        }
    
    def _validate_config(self) -> None:
        """Validate the loaded configuration structure."""
        required_keys = ["overall_threshold", "critical_modules", "quality_gates"]
        for key in required_keys:
            if key not in self._config:
                raise ValueError(f"Missing required key '{key}' in coverage thresholds configuration")
    
    def get_module_threshold(self, module_path: str) -> Tuple[float, bool]:
        """Get threshold and criticality for a specific module."""
        config = self.load_config()
        
        # Check critical modules first
        for pattern, threshold in config["critical_modules"].items():
            if self._path_matches_pattern(module_path, pattern):
                return threshold, True
        
        # Check regular module thresholds
        for pattern, threshold in config.get("module_thresholds", {}).items():
            if self._path_matches_pattern(module_path, pattern):
                return threshold, False
        
        # Default to overall threshold
        return config["overall_threshold"], False
    
    def _path_matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if a file path matches a given pattern."""
        # Convert glob-like pattern to regex
        pattern = pattern.replace("/", r"[/\\]")  # Handle both Unix and Windows paths
        pattern = pattern.replace("*", ".*")
        pattern = f"^{pattern}"
        
        # If pattern ends with /, it matches directory and all files within
        if pattern.endswith("[/\\\\]"):
            pattern = pattern[:-6] + r"[/\\].*"
        
        return bool(re.match(pattern, path))


class QualityGateLoader:
    """Loads and validates quality gate configuration."""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._config = None
    
    def load_config(self) -> Dict[str, Any]:
        """Load quality gate configuration from YAML file."""
        if self._config is None:
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f)
                self._validate_config()
            except FileNotFoundError:
                # Provide default configuration if file doesn't exist
                self._config = self._get_default_config()
                print(f"Warning: Quality gates file not found at {self.config_path}, using defaults")
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in quality gates file: {e}")
        
        return self._config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Provide default quality gate configuration based on requirements."""
        return {
            "coverage": {
                "overall_coverage_threshold": 90.0,
                "critical_module_coverage_threshold": 100.0,
                "branch_coverage_required": True,
                "fail_on_violation": True
            },
            "performance": {
                "data_loading_sla": "1s per 100MB",
                "dataframe_transformation_sla": "500ms per 1M rows",
                "benchmark_execution_timeout": 600,
                "fail_on_violation": True
            },
            "execution": {
                "test_execution_timeout": 300,
                "strict_validation": True,
                "block_merge_on_failure": True
            }
        }
    
    def _validate_config(self) -> None:
        """Validate the loaded configuration structure."""
        required_sections = ["coverage", "performance", "execution"]
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required section '{section}' in quality gates configuration")


class CoverageAnalyzer:
    """Analyzes coverage reports and validates against thresholds."""
    
    def __init__(self, threshold_loader: CoverageThresholdLoader, quality_gate_loader: QualityGateLoader):
        self.threshold_loader = threshold_loader
        self.quality_gate_loader = quality_gate_loader
    
    def analyze_coverage_report(self, coverage_file: Path) -> QualityGateResults:
        """Analyze coverage report and validate against quality gates."""
        try:
            if coverage_file.suffix == '.xml':
                return self._analyze_xml_coverage(coverage_file)
            elif coverage_file.suffix == '.json':
                return self._analyze_json_coverage(coverage_file)
            else:
                raise ValueError(f"Unsupported coverage file format: {coverage_file.suffix}")
        
        except FileNotFoundError:
            print(f"Warning: Coverage file not found at {coverage_file}, using minimal validation")
            return self._create_minimal_results()
        except Exception as e:
            print(f"Error analyzing coverage: {e}")
            return self._create_error_results(str(e))
    
    def _analyze_xml_coverage(self, coverage_file: Path) -> QualityGateResults:
        """Analyze XML coverage report (Cobertura format)."""
        tree = ET.parse(coverage_file)
        root = tree.getroot()
        
        results = QualityGateResults()
        
        # Extract overall coverage
        overall_line_rate = float(root.get('line-rate', 0.0)) * 100
        overall_branch_rate = float(root.get('branch-rate', 0.0)) * 100
        
        results.overall_coverage = overall_line_rate
        results.overall_branch_coverage = overall_branch_rate
        
        # Analyze packages and classes
        for package in root.findall('.//package'):
            package_name = package.get('name', '')
            
            for class_elem in package.findall('.//class'):
                filename = class_elem.get('filename', '')
                if not filename:
                    continue
                
                # Convert to consistent path format
                module_path = f"src/{filename}" if not filename.startswith('src/') else filename
                
                # Extract metrics
                line_rate = float(class_elem.get('line-rate', 0.0)) * 100
                branch_rate = float(class_elem.get('branch-rate', 0.0)) * 100
                
                # Count lines and branches
                lines = class_elem.findall('.//line')
                total_lines = len(lines)
                covered_lines = sum(1 for line in lines if int(line.get('hits', 0)) > 0)
                
                # Extract missing lines
                missing_lines = [int(line.get('number')) for line in lines if int(line.get('hits', 0)) == 0]
                
                # Get thresholds
                threshold, is_critical = self.threshold_loader.get_module_threshold(module_path)
                
                # Create metrics
                metrics = CoverageMetrics(
                    line_coverage=line_rate,
                    branch_coverage=branch_rate,
                    total_lines=total_lines,
                    covered_lines=covered_lines,
                    total_branches=0,  # XML doesn't always provide this
                    covered_branches=0,
                    missing_lines=missing_lines
                )
                
                # Create module report
                module_report = ModuleCoverageReport(
                    module_name=module_path,
                    metrics=metrics,
                    threshold=threshold,
                    is_critical=is_critical
                )
                
                # Check violations
                if line_rate < threshold:
                    violation = f"Module {module_path} coverage {line_rate:.1f}% below threshold {threshold:.1f}%"
                    module_report.violations.append(violation)
                    results.coverage_violations.append(violation)
                
                results.module_reports.append(module_report)
        
        # Check overall thresholds
        config = self.quality_gate_loader.load_config()
        overall_threshold = config["coverage"]["overall_coverage_threshold"]
        
        if overall_line_rate < overall_threshold:
            violation = f"Overall coverage {overall_line_rate:.1f}% below threshold {overall_threshold:.1f}%"
            results.coverage_violations.append(violation)
        
        # Check branch coverage if required
        if config["coverage"].get("branch_coverage_required", True):
            min_branch_coverage = self.threshold_loader.load_config()["quality_gates"].get("minimum_branch_coverage", 85.0)
            if overall_branch_rate < min_branch_coverage:
                violation = f"Branch coverage {overall_branch_rate:.1f}% below minimum {min_branch_coverage:.1f}%"
                results.coverage_violations.append(violation)
        
        results.passed = len(results.coverage_violations) == 0
        return results
    
    def _analyze_json_coverage(self, coverage_file: Path) -> QualityGateResults:
        """Analyze JSON coverage report."""
        with open(coverage_file, 'r', encoding='utf-8') as f:
            coverage_data = json.load(f)
        
        results = QualityGateResults()
        
        # Extract totals if available
        totals = coverage_data.get('totals', {})
        results.overall_coverage = totals.get('percent_covered', 0.0)
        results.overall_branch_coverage = totals.get('percent_covered_display', 0.0)
        
        # Analyze individual files
        files = coverage_data.get('files', {})
        for filename, file_data in files.items():
            module_path = filename
            
            # Extract summary data
            summary = file_data.get('summary', {})
            line_coverage = summary.get('percent_covered', 0.0)
            
            # Get missing lines
            missing_lines = file_data.get('missing_lines', [])
            covered_lines = summary.get('covered_lines', 0)
            total_lines = summary.get('num_statements', 0)
            
            # Get thresholds
            threshold, is_critical = self.threshold_loader.get_module_threshold(module_path)
            
            # Create metrics
            metrics = CoverageMetrics(
                line_coverage=line_coverage,
                branch_coverage=line_coverage,  # JSON format may not separate these
                total_lines=total_lines,
                covered_lines=covered_lines,
                total_branches=0,
                covered_branches=0,
                missing_lines=missing_lines
            )
            
            # Create module report
            module_report = ModuleCoverageReport(
                module_name=module_path,
                metrics=metrics,
                threshold=threshold,
                is_critical=is_critical
            )
            
            # Check violations
            if line_coverage < threshold:
                violation = f"Module {module_path} coverage {line_coverage:.1f}% below threshold {threshold:.1f}%"
                module_report.violations.append(violation)
                results.coverage_violations.append(violation)
            
            results.module_reports.append(module_report)
        
        results.passed = len(results.coverage_violations) == 0
        return results
    
    def _create_minimal_results(self) -> QualityGateResults:
        """Create minimal results when coverage file is not available."""
        results = QualityGateResults()
        results.passed = False
        results.coverage_violations.append("Coverage report file not found - unable to validate coverage")
        return results
    
    def _create_error_results(self, error_message: str) -> QualityGateResults:
        """Create error results when coverage analysis fails."""
        results = QualityGateResults()
        results.passed = False
        results.coverage_violations.append(f"Coverage analysis failed: {error_message}")
        return results


class PerformanceAnalyzer:
    """Analyzes performance benchmark results and validates against SLAs."""
    
    def __init__(self, quality_gate_loader: QualityGateLoader):
        self.quality_gate_loader = quality_gate_loader
    
    def analyze_benchmark_results(self, benchmark_file: Path) -> List[str]:
        """Analyze benchmark results and return performance violations."""
        violations = []
        
        try:
            if not benchmark_file.exists():
                print(f"Warning: Benchmark file not found at {benchmark_file}")
                return violations
            
            with open(benchmark_file, 'r', encoding='utf-8') as f:
                benchmark_data = json.load(f)
            
            config = self.quality_gate_loader.load_config()
            performance_config = config["performance"]
            
            # Analyze data loading SLA (1s per 100MB)
            data_loading_sla = self._parse_sla(performance_config["data_loading_sla"])
            
            # Analyze DataFrame transformation SLA (500ms per 1M rows)
            transformation_sla = self._parse_sla(performance_config["dataframe_transformation_sla"])
            
            # Process benchmarks
            benchmarks = benchmark_data.get('benchmarks', [])
            for benchmark in benchmarks:
                benchmark_name = benchmark.get('name', '')
                mean_time = benchmark.get('stats', {}).get('mean', 0.0)
                
                # Check data loading benchmarks
                if 'data_loading' in benchmark_name.lower() or 'load' in benchmark_name.lower():
                    # Extract data size from benchmark name or params
                    data_size_mb = self._extract_data_size(benchmark)
                    if data_size_mb > 0:
                        expected_time = data_size_mb * data_loading_sla['time_per_unit']
                        if mean_time > expected_time:
                            violations.append(
                                f"Data loading benchmark '{benchmark_name}' failed SLA: "
                                f"{mean_time:.3f}s > {expected_time:.3f}s for {data_size_mb}MB"
                            )
                
                # Check transformation benchmarks
                elif 'transform' in benchmark_name.lower() or 'dataframe' in benchmark_name.lower():
                    # Extract row count from benchmark name or params
                    row_count_millions = self._extract_row_count(benchmark)
                    if row_count_millions > 0:
                        expected_time = row_count_millions * transformation_sla['time_per_unit']
                        if mean_time > expected_time:
                            violations.append(
                                f"Transformation benchmark '{benchmark_name}' failed SLA: "
                                f"{mean_time:.3f}s > {expected_time:.3f}s for {row_count_millions}M rows"
                            )
        
        except Exception as e:
            violations.append(f"Performance analysis failed: {e}")
        
        return violations
    
    def _parse_sla(self, sla_string: str) -> Dict[str, Any]:
        """Parse SLA string like '1s per 100MB' into structured format."""
        # Extract time and unit
        match = re.match(r'(\d+(?:\.\d+)?)(\w+)\s+per\s+(\d+(?:\.\d+)?)(\w+)', sla_string)
        if not match:
            return {'time_per_unit': 1.0, 'unit': 'MB'}
        
        time_value = float(match.group(1))
        time_unit = match.group(2)
        data_value = float(match.group(3))
        data_unit = match.group(4)
        
        # Convert to standard units (seconds per MB or seconds per million rows)
        time_multiplier = {'s': 1.0, 'ms': 0.001}.get(time_unit, 1.0)
        data_multiplier = {'MB': 1.0, 'GB': 1000.0, 'M': 1.0}.get(data_unit, 1.0)
        
        time_per_unit = (time_value * time_multiplier) / (data_value * data_multiplier)
        
        return {
            'time_per_unit': time_per_unit,
            'unit': data_unit
        }
    
    def _extract_data_size(self, benchmark: Dict[str, Any]) -> float:
        """Extract data size in MB from benchmark metadata."""
        # Try to extract from parameters
        params = benchmark.get('params', {})
        if 'data_size_mb' in params:
            return float(params['data_size_mb'])
        
        # Try to extract from name
        name = benchmark.get('name', '')
        match = re.search(r'(\d+(?:\.\d+)?)mb', name.lower())
        if match:
            return float(match.group(1))
        
        # Default assumption for missing data
        return 100.0  # Assume 100MB for SLA calculation
    
    def _extract_row_count(self, benchmark: Dict[str, Any]) -> float:
        """Extract row count in millions from benchmark metadata."""
        # Try to extract from parameters
        params = benchmark.get('params', {})
        if 'row_count_millions' in params:
            return float(params['row_count_millions'])
        
        # Try to extract from name
        name = benchmark.get('name', '')
        match = re.search(r'(\d+(?:\.\d+)?)m(?:_?rows?)?', name.lower())
        if match:
            return float(match.group(1))
        
        # Default assumption for missing data
        return 1.0  # Assume 1M rows for SLA calculation


class QualityGateValidator:
    """Main class for comprehensive quality gate validation."""
    
    def __init__(self, coverage_file: Path, benchmark_file: Optional[Path] = None):
        self.coverage_file = coverage_file
        self.benchmark_file = benchmark_file
        
        # Initialize configuration loaders
        threshold_config_path = Path("tests/coverage/coverage-thresholds.json")
        quality_gate_config_path = Path("tests/coverage/quality-gates.yml")
        
        self.threshold_loader = CoverageThresholdLoader(threshold_config_path)
        self.quality_gate_loader = QualityGateLoader(quality_gate_config_path)
        
        # Initialize analyzers
        self.coverage_analyzer = CoverageAnalyzer(self.threshold_loader, self.quality_gate_loader)
        self.performance_analyzer = PerformanceAnalyzer(self.quality_gate_loader)
    
    def validate_quality_gates(self) -> QualityGateResults:
        """Perform comprehensive quality gate validation."""
        print("🔍 Starting comprehensive quality gate validation...")
        print(f"📊 Coverage file: {self.coverage_file}")
        if self.benchmark_file:
            print(f"⚡ Benchmark file: {self.benchmark_file}")
        
        # Analyze coverage
        print("\n📈 Analyzing coverage metrics...")
        results = self.coverage_analyzer.analyze_coverage_report(self.coverage_file)
        
        # Analyze performance if benchmark file is provided
        if self.benchmark_file:
            print("🚀 Analyzing performance benchmarks...")
            performance_violations = self.performance_analyzer.analyze_benchmark_results(self.benchmark_file)
            results.performance_violations.extend(performance_violations)
        
        # Update overall pass/fail status
        results.passed = (
            len(results.coverage_violations) == 0 and 
            len(results.performance_violations) == 0
        )
        
        return results
    
    def generate_report(self, results: QualityGateResults) -> str:
        """Generate comprehensive quality gate report."""
        report_lines = []
        
        # Header
        status_emoji = "✅" if results.passed else "❌"
        report_lines.append(f"{status_emoji} Quality Gate Validation Report")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Overall Status: {'PASSED' if results.passed else 'FAILED'}")
        report_lines.append("")
        
        # Coverage Summary
        report_lines.append("📊 Coverage Summary")
        report_lines.append("-" * 20)
        report_lines.append(f"Overall Line Coverage: {results.overall_coverage:.1f}%")
        report_lines.append(f"Overall Branch Coverage: {results.overall_branch_coverage:.1f}%")
        report_lines.append(f"Total Modules Analyzed: {len(results.module_reports)}")
        
        # Critical modules summary
        critical_modules = [r for r in results.module_reports if r.is_critical]
        if critical_modules:
            report_lines.append(f"Critical Modules: {len(critical_modules)}")
            for module in critical_modules:
                status = "✅" if len(module.violations) == 0 else "❌"
                report_lines.append(f"  {status} {module.module_name}: {module.metrics.line_coverage:.1f}%")
        
        report_lines.append("")
        
        # Coverage Violations
        if results.coverage_violations:
            report_lines.append("❌ Coverage Violations")
            report_lines.append("-" * 22)
            for violation in results.coverage_violations:
                report_lines.append(f"  • {violation}")
            report_lines.append("")
        
        # Performance Violations
        if results.performance_violations:
            report_lines.append("❌ Performance Violations")
            report_lines.append("-" * 25)
            for violation in results.performance_violations:
                report_lines.append(f"  • {violation}")
            report_lines.append("")
        
        # Module Details
        if results.module_reports:
            report_lines.append("📋 Module Coverage Details")
            report_lines.append("-" * 27)
            
            for module in sorted(results.module_reports, key=lambda x: x.metrics.line_coverage):
                status = "✅" if len(module.violations) == 0 else "❌"
                critical_marker = " [CRITICAL]" if module.is_critical else ""
                
                report_lines.append(
                    f"  {status} {module.module_name}{critical_marker}"
                )
                report_lines.append(
                    f"     Line Coverage: {module.metrics.line_coverage:.1f}% "
                    f"(Threshold: {module.threshold:.1f}%)"
                )
                
                if module.metrics.missing_lines and len(module.metrics.missing_lines) <= 10:
                    lines_str = ", ".join(map(str, module.metrics.missing_lines))
                    report_lines.append(f"     Missing Lines: {lines_str}")
                elif module.metrics.missing_lines:
                    report_lines.append(f"     Missing Lines: {len(module.metrics.missing_lines)} lines")
                
                if module.violations:
                    for violation in module.violations:
                        report_lines.append(f"     ⚠️  {violation}")
                
                report_lines.append("")
        
        # Quality Gate Decision
        report_lines.append("🎯 Quality Gate Decision")
        report_lines.append("-" * 24)
        
        if results.passed:
            report_lines.append("✅ ALL QUALITY GATES PASSED")
            report_lines.append("   Merge is allowed to proceed")
        else:
            report_lines.append("❌ QUALITY GATE VIOLATIONS DETECTED")
            report_lines.append("   Merge is BLOCKED until violations are resolved")
            
            # Action items
            report_lines.append("")
            report_lines.append("📝 Required Actions:")
            if results.coverage_violations:
                report_lines.append("   1. Increase test coverage for failing modules")
                report_lines.append("   2. Ensure critical modules achieve 100% coverage")
                report_lines.append("   3. Add branch coverage tests where needed")
            
            if results.performance_violations:
                report_lines.append("   4. Optimize performance for failing benchmarks")
                report_lines.append("   5. Review data loading and transformation algorithms")
        
        return "\n".join(report_lines)


def main():
    """Main entry point for coverage validation script."""
    parser = argparse.ArgumentParser(description="Validate coverage and performance quality gates")
    parser.add_argument(
        "--coverage-file", 
        type=Path, 
        default=Path("coverage.xml"),
        help="Path to coverage report file (XML or JSON)"
    )
    parser.add_argument(
        "--benchmark-file", 
        type=Path,
        help="Path to benchmark results file (JSON)"
    )
    parser.add_argument(
        "--output-report", 
        type=Path,
        help="Path to write detailed quality gate report"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize validator
        validator = QualityGateValidator(args.coverage_file, args.benchmark_file)
        
        # Run validation
        results = validator.validate_quality_gates()
        
        # Generate and display report
        report = validator.generate_report(results)
        print(report)
        
        # Write report to file if requested
        if args.output_report:
            with open(args.output_report, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n📄 Detailed report written to: {args.output_report}")
        
        # Exit with appropriate code for CI/CD integration
        exit_code = 0 if results.passed else 1
        
        if args.verbose:
            print(f"\n🚪 Exiting with code: {exit_code}")
            if exit_code == 1:
                print("   This will block the merge in CI/CD pipeline")
            else:
                print("   Quality gates passed - merge allowed")
        
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"❌ Quality gate validation failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()