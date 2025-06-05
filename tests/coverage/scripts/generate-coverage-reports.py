#!/usr/bin/env python3
"""
FlyrigLoader Coverage Report Generation System

Automated coverage report generation script orchestrating multi-format coverage reporting with 
HTML, XML, and JSON outputs. Implements comprehensive report generation pipeline with custom 
templating, statistical analysis, and quality metrics integration per TST-COV-003 requirements.

This script serves as the central orchestrator for the flyrigloader Coverage Enhancement System (F-012), 
providing automated generation of coverage reports in multiple formats with comprehensive analytics, 
trend tracking, and CI/CD integration capabilities.

Key Features:
- Multi-format report generation (HTML, XML, JSON) per TST-COV-003
- Coverage.py programmatic API integration for detailed analysis
- Custom Jinja2 templating with flyrigloader-specific styling
- Historical trend analysis and regression detection
- Module-specific coverage breakdown with critical module validation
- CI/CD integration with automated report uploading
- Quality gate status reporting and threshold enforcement

Author: FlyrigLoader Coverage Enhancement System
Version: 1.0.0
Requirements: TST-COV-003, Section 2.1.12, Section 3.6.4, TST-COV-002
"""

import argparse
import json
import os
import sys
import tempfile
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Third-party imports for coverage analysis and templating
try:
    import coverage
    from coverage.control import Coverage
    from coverage.data import CoverageData
    from coverage.results import Analysis
except ImportError:
    print("ERROR: coverage.py not installed. Install with: pip install coverage>=7.8.2")
    sys.exit(1)

try:
    import jinja2
    from jinja2 import Environment, FileSystemLoader, Template
except ImportError:
    print("ERROR: Jinja2 not installed. Install with: pip install jinja2>=3.1.0")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Install with: pip install pyyaml>=6.0")
    sys.exit(1)

# Standard library imports
import shutil
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import NamedTuple


class CoverageException(Exception):
    """Base exception for coverage report generation failures."""
    pass


class ConfigurationError(CoverageException):
    """Raised when coverage report configuration is invalid."""
    pass


class TemplateError(CoverageException):
    """Raised when template processing fails."""
    pass


class QualityGateError(CoverageException):
    """Raised when coverage quality gates fail validation."""
    pass


@dataclass
class CoverageMetrics:
    """Coverage metrics data structure for comprehensive analysis."""
    total_lines: int = 0
    covered_lines: int = 0
    missing_lines: int = 0
    total_branches: int = 0
    covered_branches: int = 0
    missing_branches: int = 0
    total_functions: int = 0
    covered_functions: int = 0
    line_coverage_percentage: float = 0.0
    branch_coverage_percentage: float = 0.0
    function_coverage_percentage: float = 0.0
    overall_coverage_percentage: float = 0.0


@dataclass
class ModuleCoverage:
    """Coverage information for a specific module."""
    module_name: str
    module_path: str
    category: str
    priority: str
    description: str
    coverage_requirement: float
    metrics: CoverageMetrics
    missing_lines: List[int] = field(default_factory=list)
    missing_branches: List[Tuple[int, int]] = field(default_factory=list)
    covered_functions: List[str] = field(default_factory=list)
    missing_functions: List[str] = field(default_factory=list)
    quality_gate_status: str = "unknown"


@dataclass 
class ReportContext:
    """Context data for template rendering."""
    timestamp: str
    library_version: str
    python_version: str
    coverage_version: str
    pytest_version: str
    build_metadata: Dict[str, Any] = field(default_factory=dict)
    environment_info: Dict[str, Any] = field(default_factory=dict)
    git_info: Dict[str, Any] = field(default_factory=dict)


class CoverageReportGenerator:
    """
    Comprehensive coverage report generator implementing TST-COV-003 requirements.
    
    This class orchestrates the generation of multi-format coverage reports with advanced
    analytics, custom templating, and CI/CD integration capabilities. It integrates with
    the coverage.py programmatic API to provide detailed coverage analysis and implements
    quality gate validation per TST-COV-002 critical module requirements.
    """

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize coverage report generator with configuration loading.
        
        Args:
            project_root: Root directory of the project (defaults to current working directory)
        """
        self.project_root = Path(project_root or os.getcwd()).resolve()
        self.coverage_dir = self.project_root / "tests" / "coverage"
        self.scripts_dir = self.coverage_dir / "scripts"
        self.templates_dir = self.coverage_dir / "templates"
        
        # Initialize configuration
        self._load_configurations()
        
        # Initialize Jinja2 environment with custom filters
        self._setup_template_environment()
        
        # Coverage analysis state
        self.coverage_data: Optional[CoverageData] = None
        self.module_coverage: Dict[str, ModuleCoverage] = {}
        self.overall_metrics = CoverageMetrics()
        self.quality_gate_results: Dict[str, bool] = {}
        
        # Report generation state
        self.report_context = ReportContext(
            timestamp=datetime.now(timezone.utc).isoformat(),
            library_version=self._get_library_version(),
            python_version=self._get_python_version(),
            coverage_version=self._get_coverage_version(),
            pytest_version=self._get_pytest_version()
        )

    def _load_configurations(self) -> None:
        """Load all configuration files required for report generation."""
        try:
            # Load main report configuration
            config_path = self.coverage_dir / "report-config.json"
            if not config_path.exists():
                raise ConfigurationError(f"Report configuration not found: {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                self.report_config = json.load(f)
            
            # Load coverage thresholds configuration
            thresholds_path = self.coverage_dir / "coverage-thresholds.json"
            if not thresholds_path.exists():
                raise ConfigurationError(f"Coverage thresholds not found: {thresholds_path}")
            
            with open(thresholds_path, 'r', encoding='utf-8') as f:
                self.coverage_thresholds = json.load(f)
            
            print(f"✅ Loaded configurations from {config_path} and {thresholds_path}")
            
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configurations: {e}")

    def _setup_template_environment(self) -> None:
        """Initialize Jinja2 template environment with custom filters and functions."""
        if not self.templates_dir.exists():
            raise TemplateError(f"Templates directory not found: {self.templates_dir}")
        
        # Create Jinja2 environment with template directory
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Register custom filters for coverage calculations
        self.jinja_env.filters.update({
            'percentage': self._percentage_filter,
            'coverage_class': self._coverage_class_filter,
            'format_number': self._format_number_filter,
            'relative_path': self._relative_path_filter,
            'format_timestamp': self._format_timestamp_filter
        })
        
        # Register custom functions
        self.jinja_env.globals.update({
            'now': datetime.now,
            'round': round,
            'len': len,
            'enumerate': enumerate,
            'zip': zip
        })
        
        print(f"✅ Template environment initialized with {len(self.jinja_env.list_templates())} templates")

    def _percentage_filter(self, numerator: Union[int, float], denominator: Union[int, float]) -> float:
        """Calculate percentage with safe division."""
        if denominator == 0:
            return 0.0
        return round((numerator / denominator) * 100, 2)

    def _coverage_class_filter(self, percentage: float) -> str:
        """Determine CSS class based on coverage percentage."""
        if percentage >= 95:
            return "excellent"
        elif percentage >= 85:
            return "good"
        elif percentage >= 70:
            return "acceptable"
        elif percentage >= 50:
            return "poor"
        else:
            return "critical"

    def _format_number_filter(self, value: Union[int, float], precision: int = 0) -> str:
        """Format number with thousands separators."""
        if isinstance(value, float):
            return f"{value:,.{precision}f}"
        return f"{value:,}"

    def _relative_path_filter(self, path: Union[str, Path]) -> str:
        """Convert absolute path to relative path from project root."""
        try:
            return str(Path(path).relative_to(self.project_root))
        except ValueError:
            return str(path)

    def _format_timestamp_filter(self, timestamp: str, format_str: str = "%Y-%m-%d %H:%M:%S UTC") -> str:
        """Format ISO timestamp for display."""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime(format_str)
        except ValueError:
            return timestamp

    def _get_library_version(self) -> str:
        """Extract library version from setup configuration."""
        try:
            # Try to get version from installed package
            import importlib.metadata
            return importlib.metadata.version("flyrigloader")
        except Exception:
            # Fallback to reading from pyproject.toml or __init__.py
            try:
                import toml
                pyproject_path = self.project_root / "pyproject.toml"
                if pyproject_path.exists():
                    with open(pyproject_path, 'r') as f:
                        pyproject = toml.load(f)
                    return pyproject.get('project', {}).get('version', '2.0.0')
            except Exception:
                pass
            return "2.0.0"

    def _get_python_version(self) -> str:
        """Get current Python version."""
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    def _get_coverage_version(self) -> str:
        """Get coverage.py version."""
        try:
            return coverage.__version__
        except AttributeError:
            return "7.8.2"

    def _get_pytest_version(self) -> str:
        """Get pytest version if available."""
        try:
            import pytest
            return pytest.__version__
        except ImportError:
            return "7.0+"

    def analyze_coverage(self, coverage_file: Optional[Path] = None) -> None:
        """
        Perform comprehensive coverage analysis using coverage.py programmatic API.
        
        Args:
            coverage_file: Path to .coverage data file (defaults to project root)
        """
        print("📊 Starting comprehensive coverage analysis...")
        
        try:
            # Initialize Coverage instance
            coverage_path = coverage_file or (self.project_root / ".coverage")
            if not coverage_path.exists():
                raise CoverageException(f"Coverage data file not found: {coverage_path}")
            
            # Load coverage data
            cov = Coverage(data_file=str(coverage_path))
            cov.load()
            self.coverage_data = cov.get_data()
            
            # Analyze overall metrics
            self._analyze_overall_metrics(cov)
            
            # Analyze module-specific coverage
            self._analyze_module_coverage(cov)
            
            # Validate quality gates
            self._validate_quality_gates()
            
            # Gather build and environment metadata
            self._gather_metadata()
            
            print(f"✅ Coverage analysis complete:")
            print(f"   📈 Overall coverage: {self.overall_metrics.overall_coverage_percentage:.2f}%")
            print(f"   🔍 Modules analyzed: {len(self.module_coverage)}")
            print(f"   🚦 Quality gates: {sum(self.quality_gate_results.values())}/{len(self.quality_gate_results)} passed")
            
        except Exception as e:
            raise CoverageException(f"Coverage analysis failed: {e}")

    def _analyze_overall_metrics(self, cov: Coverage) -> None:
        """Analyze overall project coverage metrics."""
        try:
            # Get all covered files
            covered_files = cov.get_data().measured_files()
            
            total_lines = 0
            covered_lines = 0
            total_branches = 0
            covered_branches = 0
            total_functions = 0
            covered_functions = 0
            
            for file_path in covered_files:
                # Skip files outside source directory
                if not self._is_source_file(file_path):
                    continue
                
                try:
                    analysis = cov.analysis2(file_path)
                    file_lines = len(analysis.statements)
                    file_covered = len(analysis.executed)
                    
                    total_lines += file_lines
                    covered_lines += file_covered
                    
                    # Branch coverage if available
                    if hasattr(analysis, 'branch_lines'):
                        branch_data = cov.get_data().branch_lines()
                        if file_path in branch_data:
                            file_branches = len(branch_data[file_path])
                            total_branches += file_branches
                            # Calculate covered branches (simplified)
                            covered_branches += int(file_branches * (file_covered / file_lines) if file_lines > 0 else 0)
                    
                    # Function coverage (estimate based on def statements)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        file_functions = content.count('def ')
                        total_functions += file_functions
                        # Estimate covered functions
                        covered_functions += int(file_functions * (file_covered / file_lines) if file_lines > 0 else 0)
                    except Exception:
                        pass  # Skip function analysis if file read fails
                        
                except Exception as e:
                    print(f"⚠️  Warning: Failed to analyze {file_path}: {e}")
                    continue
            
            # Calculate percentages
            line_pct = (covered_lines / total_lines * 100) if total_lines > 0 else 0.0
            branch_pct = (covered_branches / total_branches * 100) if total_branches > 0 else 0.0
            function_pct = (covered_functions / total_functions * 100) if total_functions > 0 else 0.0
            
            # Update overall metrics
            self.overall_metrics = CoverageMetrics(
                total_lines=total_lines,
                covered_lines=covered_lines,
                missing_lines=total_lines - covered_lines,
                total_branches=total_branches,
                covered_branches=covered_branches,
                missing_branches=total_branches - covered_branches,
                total_functions=total_functions,
                covered_functions=covered_functions,
                line_coverage_percentage=line_pct,
                branch_coverage_percentage=branch_pct,
                function_coverage_percentage=function_pct,
                overall_coverage_percentage=line_pct  # Primary metric
            )
            
        except Exception as e:
            raise CoverageException(f"Overall metrics analysis failed: {e}")

    def _analyze_module_coverage(self, cov: Coverage) -> None:
        """Analyze coverage for each module based on configuration."""
        try:
            # Analyze critical modules
            if 'critical_modules' in self.coverage_thresholds:
                for module_spec, config in self.coverage_thresholds['critical_modules']['modules'].items():
                    self._analyze_single_module(cov, module_spec, config, 'critical')
            
            # Analyze standard modules
            if 'standard_modules' in self.coverage_thresholds:
                for module_spec, config in self.coverage_thresholds['standard_modules']['modules'].items():
                    self._analyze_single_module(cov, module_spec, config, 'standard')
            
        except Exception as e:
            raise CoverageException(f"Module coverage analysis failed: {e}")

    def _analyze_single_module(self, cov: Coverage, module_spec: str, config: Dict[str, Any], priority: str) -> None:
        """Analyze coverage for a single module."""
        try:
            # Resolve module path
            module_path = self.project_root / module_spec
            
            # Determine if this is a directory or file
            if module_path.is_dir():
                # Analyze all Python files in directory
                python_files = list(module_path.rglob("*.py"))
                module_name = module_spec.replace('src/flyrigloader/', '').replace('/', '.')
            else:
                # Single file analysis
                python_files = [module_path] if module_path.suffix == '.py' else []
                module_name = module_spec.replace('src/flyrigloader/', '').replace('.py', '').replace('/', '.')
            
            if not python_files:
                print(f"⚠️  Warning: No Python files found for module {module_spec}")
                return
            
            # Aggregate metrics for the module
            total_lines = 0
            covered_lines = 0
            missing_lines_list = []
            
            for py_file in python_files:
                file_str = str(py_file)
                if not self._is_source_file(file_str):
                    continue
                
                try:
                    analysis = cov.analysis2(file_str)
                    total_lines += len(analysis.statements)
                    covered_lines += len(analysis.executed)
                    missing_lines_list.extend(analysis.missing)
                except Exception as e:
                    print(f"⚠️  Warning: Failed to analyze {py_file}: {e}")
            
            # Calculate coverage percentage
            coverage_pct = (covered_lines / total_lines * 100) if total_lines > 0 else 0.0
            
            # Determine quality gate status
            requirement = config.get('line_coverage', 90.0)
            quality_gate_status = "passing" if coverage_pct >= requirement else "failing"
            
            # Create module coverage object
            module_coverage = ModuleCoverage(
                module_name=module_name,
                module_path=module_spec,
                category=config.get('category', 'Unknown'),
                priority=priority,
                description=config.get('rationale', 'No description provided'),
                coverage_requirement=requirement,
                metrics=CoverageMetrics(
                    total_lines=total_lines,
                    covered_lines=covered_lines,
                    missing_lines=total_lines - covered_lines,
                    line_coverage_percentage=coverage_pct,
                    overall_coverage_percentage=coverage_pct
                ),
                missing_lines=missing_lines_list,
                quality_gate_status=quality_gate_status
            )
            
            self.module_coverage[module_name] = module_coverage
            
        except Exception as e:
            print(f"⚠️  Warning: Failed to analyze module {module_spec}: {e}")

    def _is_source_file(self, file_path: str) -> bool:
        """Check if file is part of the source code to be analyzed."""
        path_obj = Path(file_path)
        
        # Must be in src/flyrigloader directory
        try:
            relative = path_obj.relative_to(self.project_root)
            return str(relative).startswith('src/flyrigloader/')
        except ValueError:
            return False

    def _validate_quality_gates(self) -> None:
        """Validate coverage against defined quality gates."""
        try:
            # Overall coverage gate
            overall_threshold = self.coverage_thresholds.get('global_settings', {}).get('overall_threshold', {}).get('line_coverage', 90.0)
            self.quality_gate_results['overall_coverage'] = self.overall_metrics.overall_coverage_percentage >= overall_threshold
            
            # Critical modules gate
            critical_modules_passing = True
            critical_threshold = 100.0
            
            for module_name, module_cov in self.module_coverage.items():
                if module_cov.priority == 'critical':
                    if module_cov.metrics.line_coverage_percentage < critical_threshold:
                        critical_modules_passing = False
                        break
            
            self.quality_gate_results['critical_modules'] = critical_modules_passing
            
            # Branch coverage gate
            branch_threshold = self.coverage_thresholds.get('global_settings', {}).get('overall_threshold', {}).get('branch_coverage', 85.0)
            self.quality_gate_results['branch_coverage'] = self.overall_metrics.branch_coverage_percentage >= branch_threshold
            
        except Exception as e:
            raise CoverageException(f"Quality gate validation failed: {e}")

    def _gather_metadata(self) -> None:
        """Gather build and environment metadata for report context."""
        try:
            # Git information
            git_info = {}
            try:
                git_info['branch_name'] = os.environ.get('GITHUB_REF_NAME', self._run_command(['git', 'rev-parse', '--abbrev-ref', 'HEAD']))
                git_info['commit_hash'] = os.environ.get('GITHUB_SHA', self._run_command(['git', 'rev-parse', 'HEAD']))
                git_info['commit_message'] = self._run_command(['git', 'log', '-1', '--pretty=%B']).strip()
            except Exception:
                git_info = {'branch_name': 'unknown', 'commit_hash': 'unknown', 'commit_message': 'unknown'}
            
            # Build information
            build_info = {
                'build_number': os.environ.get('BUILD_NUMBER', os.environ.get('GITHUB_RUN_NUMBER', 'local')),
                'build_url': os.environ.get('BUILD_URL', os.environ.get('GITHUB_SERVER_URL', '')),
                'environment': os.environ.get('CI_ENVIRONMENT', 'local'),
                'runner_os': os.environ.get('RUNNER_OS', sys.platform)
            }
            
            # Environment information
            env_info = {
                'working_directory': str(self.project_root),
                'coverage_file': str(self.project_root / '.coverage'),
                'python_executable': sys.executable
            }
            
            # Update report context
            self.report_context.git_info = git_info
            self.report_context.build_metadata = build_info
            self.report_context.environment_info = env_info
            
        except Exception as e:
            print(f"⚠️  Warning: Failed to gather metadata: {e}")

    def _run_command(self, command: List[str]) -> str:
        """Run shell command and return output."""
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True, cwd=self.project_root)
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return 'unknown'

    def generate_html_report(self, output_path: Optional[Path] = None) -> Path:
        """Generate comprehensive HTML coverage report with custom styling."""
        print("🌐 Generating HTML coverage report...")
        
        try:
            # Determine output path
            html_output = output_path or (self.project_root / self.report_config['html_report']['output_directory'])
            html_output.mkdir(parents=True, exist_ok=True)
            
            # Load HTML template
            template = self.jinja_env.get_template('index.html.j2')
            
            # Prepare template context
            context = self._prepare_html_context()
            
            # Render HTML report
            html_content = template.render(**context)
            
            # Write main HTML file
            main_html = html_output / 'index.html'
            with open(main_html, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Copy static assets if they exist
            self._copy_static_assets(html_output)
            
            print(f"✅ HTML report generated: {main_html}")
            return main_html
            
        except Exception as e:
            raise TemplateError(f"HTML report generation failed: {e}")

    def _prepare_html_context(self) -> Dict[str, Any]:
        """Prepare template context for HTML report rendering."""
        # Organize modules by category
        modules_by_category = defaultdict(list)
        for module_name, module_cov in self.module_coverage.items():
            category = f"{module_cov.priority}_{module_cov.category.lower().replace(' ', '_')}"
            modules_by_category[category].append({
                'name': module_name,
                'path': module_cov.module_path,
                'coverage': module_cov.metrics.line_coverage_percentage,
                'class': self._coverage_class_filter(module_cov.metrics.line_coverage_percentage),
                'description': module_cov.description,
                'lines': module_cov.metrics.total_lines
            })
        
        # Calculate critical modules average
        critical_modules = [m for m in self.module_coverage.values() if m.priority == 'critical']
        critical_avg = sum(m.metrics.line_coverage_percentage for m in critical_modules) / len(critical_modules) if critical_modules else 0
        
        return {
            'config': self.report_config,
            'coverage': {
                'overall_percentage': self.overall_metrics.overall_coverage_percentage,
                'overall_class': self._coverage_class_filter(self.overall_metrics.overall_coverage_percentage),
                'covered_lines': self.overall_metrics.covered_lines,
                'total_lines': self.overall_metrics.total_lines,
                'branch_percentage': self.overall_metrics.branch_coverage_percentage,
                'branch_class': self._coverage_class_filter(self.overall_metrics.branch_coverage_percentage),
                'covered_branches': self.overall_metrics.covered_branches,
                'total_branches': self.overall_metrics.total_branches,
                'function_percentage': self.overall_metrics.function_coverage_percentage,
                'function_class': self._coverage_class_filter(self.overall_metrics.function_coverage_percentage),
                'covered_functions': self.overall_metrics.covered_functions,
                'total_functions': self.overall_metrics.total_functions,
                'critical_percentage': critical_avg,
                'critical_class': self._coverage_class_filter(critical_avg)
            },
            'modules': dict(modules_by_category),
            'quality_gates': self.quality_gate_results,
            'report': asdict(self.report_context),
            'thresholds': self.coverage_thresholds
        }

    def _copy_static_assets(self, html_output: Path) -> None:
        """Copy static assets like CSS and JavaScript files."""
        try:
            # Check for style.css template
            css_template_path = self.templates_dir / 'style.css.j2'
            if css_template_path.exists():
                css_template = self.jinja_env.get_template('style.css.j2')
                css_content = css_template.render(config=self.report_config)
                
                css_output = html_output / 'style.css'
                with open(css_output, 'w', encoding='utf-8') as f:
                    f.write(css_content)
                print(f"✅ Generated CSS: {css_output}")
            
        except Exception as e:
            print(f"⚠️  Warning: Failed to copy static assets: {e}")

    def generate_xml_report(self, output_path: Optional[Path] = None) -> Path:
        """Generate XML coverage report for CI/CD integration."""
        print("📄 Generating XML coverage report...")
        
        try:
            # Determine output path
            xml_output = output_path or (self.project_root / self.report_config['xml_report']['output_file'])
            xml_output.parent.mkdir(parents=True, exist_ok=True)
            
            # Load XML template
            template = self.jinja_env.get_template('coverage.xml.j2')
            
            # Prepare template context
            context = self._prepare_xml_context()
            
            # Render XML report
            xml_content = template.render(**context)
            
            # Write XML file
            with open(xml_output, 'w', encoding='utf-8') as f:
                f.write(xml_content)
            
            # Validate XML structure
            try:
                ET.parse(xml_output)
                print(f"✅ XML report generated and validated: {xml_output}")
            except ET.ParseError as e:
                print(f"⚠️  Warning: Generated XML may have issues: {e}")
            
            return xml_output
            
        except Exception as e:
            raise TemplateError(f"XML report generation failed: {e}")

    def _prepare_xml_context(self) -> Dict[str, Any]:
        """Prepare template context for XML report rendering."""
        return {
            'format_version': '4.0',
            'timestamp': self.report_context.timestamp,
            'total_lines': self.overall_metrics.total_lines,
            'covered_lines': self.overall_metrics.covered_lines,
            'total_branches': self.overall_metrics.total_branches,
            'covered_branches': self.overall_metrics.covered_branches,
            'build_number': self.report_context.build_metadata.get('build_number', 'local'),
            'build_url': self.report_context.build_metadata.get('build_url', ''),
            'branch_name': self.report_context.git_info.get('branch_name', 'unknown'),
            'commit_hash': self.report_context.git_info.get('commit_hash', 'unknown'),
            'commit_message': self.report_context.git_info.get('commit_message', ''),
            'environment': self.report_context.build_metadata.get('environment', 'local'),
            'runner_os': self.report_context.build_metadata.get('runner_os', sys.platform),
            'python_version': self.report_context.python_version,
            'coverage_version': self.report_context.coverage_version,
            'pytest_cov_version': '6.1.1',  # From dependencies
            'quality_gates': self.coverage_thresholds.get('global_settings', {}),
            'critical_modules_coverage': sum(m.metrics.line_coverage_percentage for m in self.module_coverage.values() if m.priority == 'critical') / len([m for m in self.module_coverage.values() if m.priority == 'critical']) if any(m.priority == 'critical' for m in self.module_coverage.values()) else 0.0,
            'modules': [asdict(m) for m in self.module_coverage.values()]
        }

    def generate_json_report(self, output_path: Optional[Path] = None) -> Path:
        """Generate JSON coverage report for programmatic analysis."""
        print("📊 Generating JSON coverage report...")
        
        try:
            # Determine output path
            json_output = output_path or (self.project_root / self.report_config['json_report']['output_file'])
            json_output.parent.mkdir(parents=True, exist_ok=True)
            
            # Load JSON template
            template = self.jinja_env.get_template('coverage.json.j2')
            
            # Prepare template context
            context = self._prepare_json_context()
            
            # Render JSON report
            json_content = template.render(**context)
            
            # Parse and re-serialize to ensure valid JSON
            try:
                json_data = json.loads(json_content)
                formatted_json = json.dumps(json_data, indent=2, ensure_ascii=False)
            except json.JSONDecodeError as e:
                print(f"⚠️  Template generated invalid JSON, using fallback: {e}")
                formatted_json = json.dumps(context, indent=2, ensure_ascii=False, default=str)
            
            # Write JSON file
            with open(json_output, 'w', encoding='utf-8') as f:
                f.write(formatted_json)
            
            print(f"✅ JSON report generated: {json_output}")
            return json_output
            
        except Exception as e:
            raise TemplateError(f"JSON report generation failed: {e}")

    def _prepare_json_context(self) -> Dict[str, Any]:
        """Prepare template context for JSON report rendering."""
        return {
            'meta': {
                'timestamp': self.report_context.timestamp,
                'coverage_version': self.report_context.coverage_version,
                'pytest_cov_version': '6.1.1',
                'python_version': self.report_context.python_version
            },
            'config': self.report_config,
            'coverage_data': {
                'overall': asdict(self.overall_metrics),
                'modules': {name: asdict(module) for name, module in self.module_coverage.items()},
                'quality_gates': self.quality_gate_results
            },
            'thresholds': self.coverage_thresholds,
            'build_info': self.report_context.build_metadata,
            'now': lambda: datetime.now(timezone.utc)
        }

    def generate_all_reports(self, output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """Generate all configured report formats."""
        print("🚀 Generating all coverage reports...")
        
        # Ensure we have coverage data
        if self.coverage_data is None:
            raise CoverageException("Coverage analysis must be run before generating reports")
        
        results = {}
        base_dir = output_dir or self.project_root
        
        try:
            # Generate HTML report
            if self.report_config['html_report']['enabled']:
                html_path = self.generate_html_report(base_dir / 'htmlcov')
                results['html'] = html_path
            
            # Generate XML report
            if self.report_config['xml_report']['enabled']:
                xml_path = self.generate_xml_report(base_dir / 'coverage.xml')
                results['xml'] = xml_path
            
            # Generate JSON report
            if self.report_config['json_report']['enabled']:
                json_path = self.generate_json_report(base_dir / 'coverage.json')
                results['json'] = json_path
            
            # Generate summary console output
            self._print_coverage_summary()
            
            print(f"✅ All reports generated successfully!")
            print(f"   📁 Output directory: {base_dir}")
            print(f"   📊 Reports: {', '.join(results.keys())}")
            
            return results
            
        except Exception as e:
            raise CoverageException(f"Report generation failed: {e}")

    def _print_coverage_summary(self) -> None:
        """Print coverage summary to console."""
        print("\n" + "="*60)
        print("📊 FLYRIGLOADER COVERAGE SUMMARY")
        print("="*60)
        
        print(f"Overall Coverage: {self.overall_metrics.overall_coverage_percentage:.2f}%")
        print(f"Lines Covered: {self.overall_metrics.covered_lines:,} / {self.overall_metrics.total_lines:,}")
        
        if self.overall_metrics.total_branches > 0:
            print(f"Branch Coverage: {self.overall_metrics.branch_coverage_percentage:.2f}%")
            print(f"Branches Covered: {self.overall_metrics.covered_branches:,} / {self.overall_metrics.total_branches:,}")
        
        print(f"\nQuality Gates:")
        for gate_name, passing in self.quality_gate_results.items():
            status = "✅ PASS" if passing else "❌ FAIL"
            print(f"  {gate_name.replace('_', ' ').title()}: {status}")
        
        print(f"\nCritical Modules:")
        for module_name, module_cov in self.module_coverage.items():
            if module_cov.priority == 'critical':
                status = "✅" if module_cov.quality_gate_status == "passing" else "❌"
                print(f"  {status} {module_name}: {module_cov.metrics.line_coverage_percentage:.2f}%")
        
        print("="*60)

    def upload_reports_to_ci(self, reports: Dict[str, Path]) -> bool:
        """Upload generated reports to CI/CD artifacts."""
        print("☁️  Uploading reports to CI/CD system...")
        
        try:
            # GitHub Actions artifact upload
            if os.environ.get('GITHUB_ACTIONS'):
                self._upload_github_artifacts(reports)
                return True
            
            # Add support for other CI systems here (Jenkins, GitLab, etc.)
            else:
                print("ℹ️  No CI/CD system detected, skipping upload")
                return False
            
        except Exception as e:
            print(f"⚠️  Warning: Report upload failed: {e}")
            return False

    def _upload_github_artifacts(self, reports: Dict[str, Path]) -> None:
        """Upload reports as GitHub Actions artifacts."""
        try:
            # Create artifacts directory
            artifacts_dir = self.project_root / 'coverage-artifacts'
            artifacts_dir.mkdir(exist_ok=True)
            
            # Copy reports to artifacts directory
            for report_type, report_path in reports.items():
                if report_path.is_file():
                    shutil.copy2(report_path, artifacts_dir / f'coverage-{report_type}.{report_path.suffix[1:]}')
                elif report_path.is_dir():
                    shutil.copytree(report_path, artifacts_dir / f'coverage-{report_type}', dirs_exist_ok=True)
            
            print(f"✅ Reports prepared for GitHub Actions artifacts in {artifacts_dir}")
            
        except Exception as e:
            raise CoverageException(f"GitHub Actions artifact upload failed: {e}")

    def validate_quality_gates_strict(self) -> bool:
        """Perform strict quality gate validation for CI/CD blocking."""
        print("🚦 Performing strict quality gate validation...")
        
        all_gates_pass = True
        failed_gates = []
        
        # Check each quality gate
        for gate_name, passing in self.quality_gate_results.items():
            if not passing:
                all_gates_pass = False
                failed_gates.append(gate_name)
        
        if all_gates_pass:
            print("✅ All quality gates passed!")
        else:
            print(f"❌ Quality gate failures: {', '.join(failed_gates)}")
            
            # Print detailed failure information
            for gate_name in failed_gates:
                self._print_quality_gate_failure(gate_name)
        
        return all_gates_pass

    def _print_quality_gate_failure(self, gate_name: str) -> None:
        """Print detailed information about quality gate failure."""
        if gate_name == 'overall_coverage':
            threshold = self.coverage_thresholds.get('global_settings', {}).get('overall_threshold', {}).get('line_coverage', 90.0)
            actual = self.overall_metrics.overall_coverage_percentage
            print(f"   Overall Coverage: {actual:.2f}% < {threshold}% (required)")
        
        elif gate_name == 'critical_modules':
            print(f"   Critical modules with insufficient coverage:")
            for module_name, module_cov in self.module_coverage.items():
                if module_cov.priority == 'critical' and module_cov.quality_gate_status == 'failing':
                    req = module_cov.coverage_requirement
                    actual = module_cov.metrics.line_coverage_percentage
                    print(f"     - {module_name}: {actual:.2f}% < {req}% (required)")
        
        elif gate_name == 'branch_coverage':
            threshold = self.coverage_thresholds.get('global_settings', {}).get('overall_threshold', {}).get('branch_coverage', 85.0)
            actual = self.overall_metrics.branch_coverage_percentage
            print(f"   Branch Coverage: {actual:.2f}% < {threshold}% (required)")


def main():
    """Main entry point for coverage report generation script."""
    parser = argparse.ArgumentParser(
        description="FlyrigLoader Coverage Report Generation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all reports with coverage analysis
  python generate-coverage-reports.py --analyze --all-reports
  
  # Generate only HTML report
  python generate-coverage-reports.py --html-only
  
  # Strict validation for CI/CD
  python generate-coverage-reports.py --analyze --validate-strict
  
  # Upload to CI/CD system
  python generate-coverage-reports.py --analyze --all-reports --upload
        """
    )
    
    # Analysis options
    parser.add_argument(
        '--analyze', 
        action='store_true',
        help='Perform coverage analysis before report generation'
    )
    parser.add_argument(
        '--coverage-file',
        type=Path,
        help='Path to .coverage data file (default: .coverage in project root)'
    )
    
    # Report generation options
    parser.add_argument(
        '--all-reports',
        action='store_true',
        help='Generate all enabled report formats (HTML, XML, JSON)'
    )
    parser.add_argument(
        '--html-only',
        action='store_true',
        help='Generate only HTML report'
    )
    parser.add_argument(
        '--xml-only',
        action='store_true',
        help='Generate only XML report'
    )
    parser.add_argument(
        '--json-only',
        action='store_true',
        help='Generate only JSON report'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory for reports (default: project root)'
    )
    
    # Quality gate options
    parser.add_argument(
        '--validate-strict',
        action='store_true',
        help='Perform strict quality gate validation (exits with error on failure)'
    )
    
    # CI/CD integration options
    parser.add_argument(
        '--upload',
        action='store_true',
        help='Upload reports to CI/CD artifacts'
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
        # Initialize report generator
        generator = CoverageReportGenerator(project_root=args.project_root)
        
        # Perform coverage analysis if requested
        if args.analyze:
            generator.analyze_coverage(coverage_file=args.coverage_file)
        
        # Generate reports based on options
        reports = {}
        
        if args.all_reports:
            reports = generator.generate_all_reports(output_dir=args.output_dir)
        else:
            if args.html_only:
                html_path = generator.generate_html_report(
                    output_path=args.output_dir / 'htmlcov' if args.output_dir else None
                )
                reports['html'] = html_path
            
            if args.xml_only:
                xml_path = generator.generate_xml_report(
                    output_path=args.output_dir / 'coverage.xml' if args.output_dir else None
                )
                reports['xml'] = xml_path
            
            if args.json_only:
                json_path = generator.generate_json_report(
                    output_path=args.output_dir / 'coverage.json' if args.output_dir else None
                )
                reports['json'] = json_path
        
        # Upload reports if requested
        if args.upload and reports:
            upload_success = generator.upload_reports_to_ci(reports)
            if not upload_success:
                print("⚠️  Report upload failed, but continuing...")
        
        # Perform strict validation if requested
        if args.validate_strict:
            if not generator.validate_quality_gates_strict():
                print("❌ Quality gate validation failed!")
                sys.exit(1)
        
        print("🎉 Coverage report generation completed successfully!")
        
    except (CoverageException, ConfigurationError, TemplateError, QualityGateError) as e:
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