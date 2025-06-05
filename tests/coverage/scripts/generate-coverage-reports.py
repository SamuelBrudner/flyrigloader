#!/usr/bin/env python3
"""
FlyrigLoader Coverage Report Generation Script

Comprehensive coverage report generation pipeline orchestrating multi-format coverage
reporting with HTML, XML, and JSON outputs. Implements comprehensive report generation
pipeline with custom templating, statistical analysis, and quality metrics integration
per TST-COV-003 requirements.

This script serves as the central orchestrator for the flyrigloader test suite coverage
enhancement system, providing:

- Multi-format coverage report generation (HTML, XML, JSON)
- Custom Jinja2 templating with flyrigloader-specific styling
- Statistical analysis and trend tracking capabilities
- CI/CD integration with automated quality gate enforcement
- Module-specific coverage breakdown with critical module validation
- Performance SLA integration and benchmark correlation
- Historical coverage comparison and regression detection

Requirements Implementation:
- TST-COV-003: Generate coverage reports in XML, JSON, and HTML formats
- Section 2.1.12: Coverage Enhancement System with detailed reporting and visualization
- Section 3.6.4: Quality metrics dashboard integration with coverage trend tracking
- TST-COV-002: Achieve 100% coverage for critical data loading and validation modules

Author: FlyrigLoader Test Suite Enhancement Team
Created: 2024-12-19
License: MIT
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports for coverage analysis and templating
try:
    import coverage
    from coverage.data import CoverageData
    from coverage.results import Analysis
except ImportError as e:
    print(f"ERROR: Coverage.py not available: {e}")
    print("Install with: pip install coverage>=7.0.0")
    sys.exit(1)

try:
    import jinja2
    from jinja2 import Environment, FileSystemLoader, select_autoescape
except ImportError as e:
    print(f"ERROR: Jinja2 not available: {e}")
    print("Install with: pip install jinja2>=3.0.0")
    sys.exit(1)

# Optional imports for enhanced functionality
try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class CoverageReportGenerator:
    """
    Comprehensive coverage report generation engine implementing multi-format
    coverage reporting with advanced templating, statistical analysis, and
    quality metrics integration.
    
    This class orchestrates the complete coverage reporting pipeline including:
    - Coverage data analysis using coverage.py API
    - Custom template rendering with Jinja2
    - Statistical metrics calculation and trend analysis
    - Quality gate validation and threshold enforcement
    - CI/CD integration and artifact generation
    """

    def __init__(self, 
                 coverage_data_file: Optional[str] = None,
                 config_file: Optional[str] = None,
                 thresholds_file: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 verbose: bool = False):
        """
        Initialize the coverage report generator with configuration and data sources.
        
        Args:
            coverage_data_file: Path to coverage data file (.coverage)
            config_file: Path to report configuration JSON file
            thresholds_file: Path to coverage thresholds JSON file
            output_dir: Base output directory for generated reports
            verbose: Enable verbose logging output
        """
        self.verbose = verbose
        self.setup_logging()
        
        # Initialize paths and configuration
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.coverage_dir = self.project_root / "tests" / "coverage"
        self.templates_dir = self.coverage_dir / "templates"
        self.scripts_dir = self.coverage_dir / "scripts"
        
        # Set default paths if not provided
        self.coverage_data_file = coverage_data_file or str(self.project_root / ".coverage")
        self.config_file = config_file or str(self.coverage_dir / "report-config.json")
        self.thresholds_file = thresholds_file or str(self.coverage_dir / "coverage-thresholds.json")
        self.output_dir = Path(output_dir) if output_dir else self.project_root / "htmlcov"
        
        # Initialize configuration and data
        self.config: Dict[str, Any] = {}
        self.thresholds: Dict[str, Any] = {}
        self.coverage_obj: Optional[coverage.Coverage] = None
        self.coverage_data: Optional[CoverageData] = None
        
        # Initialize Jinja2 environment
        self.jinja_env: Optional[Environment] = None
        
        # Runtime statistics
        self.start_time = time.time()
        self.generation_stats: Dict[str, Any] = {
            'start_time': datetime.now(timezone.utc),
            'reports_generated': [],
            'warnings': [],
            'errors': []
        }
        
        self.logger.info(f"Initialized CoverageReportGenerator")
        self.logger.info(f"Project root: {self.project_root}")
        self.logger.info(f"Coverage data file: {self.coverage_data_file}")
        self.logger.info(f"Output directory: {self.output_dir}")

    def setup_logging(self) -> None:
        """Configure comprehensive logging for coverage report generation."""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(
                    self.project_root / "tests" / "coverage" / "coverage-reports.log",
                    mode='a'
                )
            ]
        )
        self.logger = logging.getLogger("flyrigloader.coverage.generator")

    def load_configuration(self) -> None:
        """
        Load comprehensive configuration from JSON files including report
        formatting, thresholds, and template settings.
        
        Raises:
            FileNotFoundError: If configuration files are missing
            json.JSONDecodeError: If configuration files are malformed
        """
        self.logger.info("Loading configuration files...")
        
        # Load report configuration
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            self.logger.info(f"Loaded report configuration from {self.config_file}")
        except FileNotFoundError:
            self.logger.error(f"Report configuration file not found: {self.config_file}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in report config: {e}")
            raise

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

        # Validate configuration structure
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate that required configuration keys are present and properly formatted."""
        required_config_keys = ['metadata', 'report_formats', 'quality_gates']
        for key in required_config_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")

        required_threshold_keys = ['global_configuration', 'module_thresholds']
        for key in required_threshold_keys:
            if key not in self.thresholds:
                raise ValueError(f"Missing required threshold key: {key}")

        self.logger.info("Configuration validation completed successfully")

    def load_coverage_data(self) -> None:
        """
        Load and analyze coverage data using coverage.py API for comprehensive
        coverage metrics extraction and module-level analysis.
        
        Raises:
            FileNotFoundError: If coverage data file is missing
            coverage.CoverageException: If coverage data is corrupted
        """
        self.logger.info(f"Loading coverage data from {self.coverage_data_file}")
        
        if not os.path.exists(self.coverage_data_file):
            raise FileNotFoundError(f"Coverage data file not found: {self.coverage_data_file}")

        try:
            # Initialize coverage object and load data
            self.coverage_obj = coverage.Coverage(data_file=self.coverage_data_file)
            self.coverage_obj.load()
            self.coverage_data = self.coverage_obj.get_data()
            
            # Log basic coverage statistics
            measured_files = self.coverage_data.measured_files()
            self.logger.info(f"Loaded coverage data for {len(measured_files)} files")
            
            if self.verbose:
                for file_path in sorted(measured_files):
                    self.logger.debug(f"Coverage data available for: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load coverage data: {e}")
            raise

    def setup_jinja_environment(self) -> None:
        """
        Initialize Jinja2 templating environment with custom filters and functions
        for flyrigloader-specific coverage report generation.
        """
        self.logger.info("Setting up Jinja2 templating environment")
        
        # Ensure templates directory exists
        if not self.templates_dir.exists():
            self.logger.warning(f"Templates directory not found: {self.templates_dir}")
            self.templates_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Jinja2 environment with security settings
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )

        # Add custom filters for coverage reporting
        self.jinja_env.filters.update({
            'percentage': self._filter_percentage,
            'datetime_format': self._filter_datetime,
            'file_size_format': self._filter_file_size,
            'module_category': self._filter_module_category,
            'coverage_status': self._filter_coverage_status,
            'format_number': self._filter_format_number
        })

        # Add custom global functions
        self.jinja_env.globals.update({
            'get_git_info': self._get_git_info,
            'get_system_info': self._get_system_info,
            'calculate_trend': self._calculate_coverage_trend,
            'get_critical_modules': self._get_critical_modules
        })

        self.logger.info("Jinja2 environment configured with custom filters and functions")

    def _filter_percentage(self, value: Union[int, float], precision: int = 2) -> str:
        """Format numeric value as percentage with specified precision."""
        if value is None:
            return "N/A"
        return f"{float(value):.{precision}f}%"

    def _filter_datetime(self, dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S UTC") -> str:
        """Format datetime object with specified format string."""
        if dt is None:
            return "N/A"
        return dt.strftime(format_str)

    def _filter_file_size_format(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes is None:
            return "N/A"
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def _filter_module_category(self, module_path: str) -> str:
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

    def _filter_coverage_status(self, coverage_pct: float, module_path: str) -> str:
        """Determine coverage status based on thresholds and module criticality."""
        category = self._filter_module_category(module_path)
        threshold_config = self.thresholds.get('module_thresholds', {})
        
        if category in ['api_layer', 'configuration_system', 'discovery_engine', 'io_pipeline']:
            threshold = 100.0
        elif category == 'utilities':
            threshold = 95.0
        elif category == 'initialization_modules':
            threshold = 85.0
        else:
            threshold = self.thresholds.get('global_configuration', {}).get('overall_threshold', 90.0)

        if coverage_pct >= threshold:
            return 'excellent'
        elif coverage_pct >= threshold - 5:
            return 'good'
        elif coverage_pct >= threshold - 10:
            return 'warning'
        else:
            return 'critical'

    def _filter_format_number(self, value: Union[int, float], precision: int = 0) -> str:
        """Format numeric value with thousands separators."""
        if value is None:
            return "N/A"
        if precision == 0:
            return f"{int(value):,}"
        else:
            return f"{float(value):,.{precision}f}"

    def _get_git_info(self) -> Dict[str, Any]:
        """Extract Git repository information for build metadata."""
        git_info = {
            'available': False,
            'branch': 'unknown',
            'commit_hash': 'unknown',
            'commit_message': 'unknown',
            'commit_timestamp': 'unknown',
            'is_dirty': False
        }

        if not GIT_AVAILABLE:
            return git_info

        try:
            repo = git.Repo(self.project_root)
            git_info.update({
                'available': True,
                'branch': repo.active_branch.name,
                'commit_hash': repo.head.commit.hexsha[:8],
                'commit_message': repo.head.commit.message.strip(),
                'commit_timestamp': datetime.fromtimestamp(
                    repo.head.commit.committed_date, tz=timezone.utc
                ).isoformat(),
                'is_dirty': repo.is_dirty()
            })
        except Exception as e:
            self.logger.warning(f"Failed to extract Git information: {e}")

        return git_info

    def _get_system_info(self) -> Dict[str, Any]:
        """Extract system information for build metadata."""
        system_info = {
            'python_version': sys.version.split()[0],
            'platform': sys.platform,
            'coverage_version': coverage.__version__ if hasattr(coverage, '__version__') else 'unknown'
        }

        if PSUTIL_AVAILABLE:
            try:
                system_info.update({
                    'cpu_count': psutil.cpu_count(),
                    'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                    'disk_usage_pct': psutil.disk_usage('/').percent
                })
            except Exception as e:
                self.logger.warning(f"Failed to extract system information: {e}")

        # Add environment variables for CI/CD integration
        ci_vars = ['BUILD_NUMBER', 'BUILD_URL', 'BRANCH_NAME', 'COMMIT_SHA', 
                   'PR_NUMBER', 'CI_ENVIRONMENT', 'RUNNER_OS']
        for var in ci_vars:
            system_info[var.lower()] = os.environ.get(var, 'unknown')

        return system_info

    def _calculate_coverage_trend(self, current_coverage: float) -> Dict[str, Any]:
        """Calculate coverage trend analysis with historical comparison."""
        # Placeholder for trend calculation - would integrate with historical data
        trend_info = {
            'current_coverage': current_coverage,
            'previous_coverage': None,
            'trend_direction': 'stable',
            'trend_percentage': 0.0,
            'baseline_coverage': self.thresholds.get('global_configuration', {}).get('overall_threshold', 90.0)
        }

        # In a full implementation, this would read historical coverage data
        # and calculate meaningful trends
        return trend_info

    def _get_critical_modules(self) -> List[str]:
        """Get list of critical modules requiring 100% coverage."""
        critical_modules = []
        module_thresholds = self.thresholds.get('module_thresholds', {})
        
        for category, config in module_thresholds.items():
            if config.get('line_threshold', 0) == 100.0:
                modules = config.get('modules', {})
                critical_modules.extend(modules.keys())
        
        return critical_modules

    def analyze_coverage_data(self) -> Dict[str, Any]:
        """
        Perform comprehensive coverage analysis including module-level
        breakdown, quality metrics, and statistical analysis.
        
        Returns:
            Dictionary containing comprehensive coverage analysis results
        """
        self.logger.info("Performing comprehensive coverage analysis")
        
        if not self.coverage_obj or not self.coverage_data:
            raise RuntimeError("Coverage data not loaded. Call load_coverage_data() first.")

        analysis_start = time.time()
        
        # Initialize analysis results
        analysis_results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'summary': {},
            'modules': {},
            'quality_gates': {},
            'statistics': {},
            'critical_modules': {},
            'warnings': [],
            'errors': []
        }

        # Get all measured files
        measured_files = list(self.coverage_data.measured_files())
        src_files = [f for f in measured_files if 'src/flyrigloader' in f and not f.endswith('__pycache__')]
        
        self.logger.info(f"Analyzing {len(src_files)} source files")

        # Initialize summary statistics
        total_statements = 0
        total_missing = 0
        total_branches = 0
        total_missing_branches = 0
        
        # Analyze each source file
        for file_path in src_files:
            try:
                # Get coverage analysis for file
                analysis = self.coverage_obj.analysis2(file_path)
                
                # Extract coverage metrics
                statements = len(analysis.statements)
                missing = len(analysis.missing)
                covered = statements - missing
                coverage_pct = (covered / statements * 100) if statements > 0 else 0.0

                # Branch coverage analysis
                branches = len(analysis.branch_lines) if hasattr(analysis, 'branch_lines') else 0
                missing_branches = 0
                if hasattr(analysis, 'missing_branch_lines'):
                    missing_branches = len(analysis.missing_branch_lines)
                
                branch_coverage_pct = 0.0
                if branches > 0:
                    covered_branches = branches - missing_branches
                    branch_coverage_pct = (covered_branches / branches * 100)

                # Store file-level analysis
                relative_path = os.path.relpath(file_path, self.project_root)
                analysis_results['modules'][relative_path] = {
                    'file_path': file_path,
                    'relative_path': relative_path,
                    'statements': statements,
                    'missing': missing,
                    'covered': covered,
                    'coverage_percentage': coverage_pct,
                    'branches': branches,
                    'missing_branches': missing_branches,
                    'branch_coverage_percentage': branch_coverage_pct,
                    'missing_lines': list(analysis.missing),
                    'category': self._filter_module_category(relative_path),
                    'status': self._filter_coverage_status(coverage_pct, relative_path),
                    'execution_counts': getattr(analysis, 'line_counts', {})
                }

                # Accumulate totals
                total_statements += statements
                total_missing += missing
                total_branches += branches
                total_missing_branches += missing_branches

            except Exception as e:
                self.logger.error(f"Failed to analyze {file_path}: {e}")
                analysis_results['errors'].append(f"Analysis failed for {file_path}: {e}")

        # Calculate summary statistics
        overall_coverage = ((total_statements - total_missing) / total_statements * 100) if total_statements > 0 else 0.0
        branch_coverage = ((total_branches - total_missing_branches) / total_branches * 100) if total_branches > 0 else 0.0

        analysis_results['summary'] = {
            'total_files': len(src_files),
            'total_statements': total_statements,
            'covered_statements': total_statements - total_missing,
            'missing_statements': total_missing,
            'overall_coverage_percentage': overall_coverage,
            'total_branches': total_branches,
            'covered_branches': total_branches - total_missing_branches,
            'missing_branches': total_missing_branches,
            'branch_coverage_percentage': branch_coverage,
            'analysis_duration': time.time() - analysis_start
        }

        # Perform quality gate validation
        analysis_results['quality_gates'] = self._validate_quality_gates(analysis_results)
        
        # Generate statistics and critical module analysis
        analysis_results['statistics'] = self._generate_coverage_statistics(analysis_results)
        analysis_results['critical_modules'] = self._analyze_critical_modules(analysis_results)

        self.logger.info(f"Coverage analysis completed in {analysis_results['summary']['analysis_duration']:.2f} seconds")
        self.logger.info(f"Overall coverage: {overall_coverage:.2f}%")
        self.logger.info(f"Branch coverage: {branch_coverage:.2f}%")

        return analysis_results

    def _validate_quality_gates(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate coverage against quality gate thresholds."""
        quality_gates = {
            'overall_threshold': {
                'threshold': self.thresholds['global_configuration']['overall_threshold'],
                'actual': analysis_results['summary']['overall_coverage_percentage'],
                'passed': False,
                'status': 'FAIL'
            },
            'branch_threshold': {
                'threshold': self.thresholds['global_configuration'].get('branch_threshold', 90.0),
                'actual': analysis_results['summary']['branch_coverage_percentage'],
                'passed': False,
                'status': 'FAIL'
            },
            'critical_modules': {
                'modules_checked': 0,
                'modules_passed': 0,
                'passed': False,
                'status': 'FAIL',
                'failures': []
            }
        }

        # Check overall threshold
        if analysis_results['summary']['overall_coverage_percentage'] >= quality_gates['overall_threshold']['threshold']:
            quality_gates['overall_threshold']['passed'] = True
            quality_gates['overall_threshold']['status'] = 'PASS'

        # Check branch threshold
        if analysis_results['summary']['branch_coverage_percentage'] >= quality_gates['branch_threshold']['threshold']:
            quality_gates['branch_threshold']['passed'] = True
            quality_gates['branch_threshold']['status'] = 'PASS'

        # Check critical modules
        critical_modules = self._get_critical_modules()
        modules_passed = 0
        
        for module_path in critical_modules:
            relative_path = os.path.relpath(module_path, self.project_root)
            if relative_path in analysis_results['modules']:
                module_coverage = analysis_results['modules'][relative_path]['coverage_percentage']
                if module_coverage >= 100.0:
                    modules_passed += 1
                else:
                    quality_gates['critical_modules']['failures'].append({
                        'module': relative_path,
                        'required': 100.0,
                        'actual': module_coverage
                    })

        quality_gates['critical_modules']['modules_checked'] = len(critical_modules)
        quality_gates['critical_modules']['modules_passed'] = modules_passed
        quality_gates['critical_modules']['passed'] = modules_passed == len(critical_modules)
        quality_gates['critical_modules']['status'] = 'PASS' if quality_gates['critical_modules']['passed'] else 'FAIL'

        return quality_gates

    def _generate_coverage_statistics(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed coverage statistics and metrics."""
        modules = analysis_results['modules']
        
        if not modules:
            return {}

        coverages = [m['coverage_percentage'] for m in modules.values()]
        
        statistics = {
            'mean_coverage': sum(coverages) / len(coverages),
            'median_coverage': sorted(coverages)[len(coverages) // 2],
            'min_coverage': min(coverages),
            'max_coverage': max(coverages),
            'std_deviation': 0.0,
            'modules_above_90': len([c for c in coverages if c >= 90.0]),
            'modules_below_90': len([c for c in coverages if c < 90.0]),
            'perfect_coverage_modules': len([c for c in coverages if c == 100.0])
        }

        # Calculate standard deviation
        mean = statistics['mean_coverage']
        variance = sum((c - mean) ** 2 for c in coverages) / len(coverages)
        statistics['std_deviation'] = variance ** 0.5

        return statistics

    def _analyze_critical_modules(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze coverage status of critical modules requiring 100% coverage."""
        critical_analysis = {
            'api_layer': [],
            'configuration_system': [],
            'discovery_engine': [],
            'io_pipeline': [],
            'total_critical': 0,
            'passing_critical': 0,
            'failing_critical': 0
        }

        for module_path, module_data in analysis_results['modules'].items():
            category = module_data['category']
            
            if category in critical_analysis and category != 'total_critical':
                critical_info = {
                    'module': module_path,
                    'coverage': module_data['coverage_percentage'],
                    'status': 'PASS' if module_data['coverage_percentage'] >= 100.0 else 'FAIL',
                    'missing_lines': len(module_data['missing_lines'])
                }
                critical_analysis[category].append(critical_info)
                critical_analysis['total_critical'] += 1
                
                if critical_info['status'] == 'PASS':
                    critical_analysis['passing_critical'] += 1
                else:
                    critical_analysis['failing_critical'] += 1

        return critical_analysis

    def generate_html_report(self, analysis_results: Dict[str, Any]) -> bool:
        """
        Generate comprehensive HTML coverage report with custom styling,
        interactive navigation, and flyrigloader-specific organization.
        
        Args:
            analysis_results: Coverage analysis results from analyze_coverage_data()
            
        Returns:
            True if report generation succeeded, False otherwise
        """
        self.logger.info("Generating HTML coverage report")
        
        try:
            # Ensure output directory exists
            html_output_dir = self.output_dir
            html_output_dir.mkdir(parents=True, exist_ok=True)

            # Prepare template context
            template_context = {
                'config': self.config,
                'thresholds': self.thresholds,
                'analysis': analysis_results,
                'generation_time': datetime.now(timezone.utc),
                'git_info': self._get_git_info(),
                'system_info': self._get_system_info(),
                'trend_info': self._calculate_coverage_trend(
                    analysis_results['summary']['overall_coverage_percentage']
                )
            }

            # Generate main index.html
            if self.templates_dir.joinpath('index.html.j2').exists():
                template = self.jinja_env.get_template('index.html.j2')
                html_content = template.render(**template_context)
                
                with open(html_output_dir / 'index.html', 'w', encoding='utf-8') as f:
                    f.write(html_content)
                self.logger.info(f"Generated main HTML report: {html_output_dir / 'index.html'}")
            else:
                self.logger.warning("index.html.j2 template not found, generating basic HTML report")
                self._generate_basic_html_report(html_output_dir, template_context)

            # Generate individual module reports
            if self.templates_dir.joinpath('pyfile.html.j2').exists():
                module_template = self.jinja_env.get_template('pyfile.html.j2')
                
                for module_path, module_data in analysis_results['modules'].items():
                    module_context = {**template_context, 'current_module': module_data}
                    module_html = module_template.render(**module_context)
                    
                    # Create module-specific HTML file
                    safe_module_name = module_path.replace('/', '_').replace('.py', '.html')
                    module_output_path = html_output_dir / safe_module_name
                    
                    with open(module_output_path, 'w', encoding='utf-8') as f:
                        f.write(module_html)

            # Copy or generate CSS styles
            if self.templates_dir.joinpath('style.css.j2').exists():
                style_template = self.jinja_env.get_template('style.css.j2')
                css_content = style_template.render(**template_context)
                
                with open(html_output_dir / 'style.css', 'w', encoding='utf-8') as f:
                    f.write(css_content)
            else:
                self._generate_basic_css(html_output_dir)

            self.generation_stats['reports_generated'].append({
                'format': 'HTML',
                'output_path': str(html_output_dir),
                'files_generated': len(list(html_output_dir.glob('*.html'))) + 1  # +1 for CSS
            })

            return True

        except Exception as e:
            self.logger.error(f"Failed to generate HTML report: {e}")
            self.generation_stats['errors'].append(f"HTML generation failed: {e}")
            return False

    def _generate_basic_html_report(self, output_dir: Path, context: Dict[str, Any]) -> None:
        """Generate a basic HTML report when templates are not available."""
        analysis = context['analysis']
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlyrigLoader Coverage Report</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <header>
        <h1>FlyrigLoader Test Coverage Report</h1>
        <p>Generated on {context['generation_time'].strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
    </header>
    
    <main>
        <section class="summary">
            <h2>Coverage Summary</h2>
            <div class="metrics">
                <div class="metric">
                    <span class="label">Overall Coverage:</span>
                    <span class="value">{analysis['summary']['overall_coverage_percentage']:.2f}%</span>
                </div>
                <div class="metric">
                    <span class="label">Branch Coverage:</span>
                    <span class="value">{analysis['summary']['branch_coverage_percentage']:.2f}%</span>
                </div>
                <div class="metric">
                    <span class="label">Total Files:</span>
                    <span class="value">{analysis['summary']['total_files']}</span>
                </div>
                <div class="metric">
                    <span class="label">Total Statements:</span>
                    <span class="value">{analysis['summary']['total_statements']:,}</span>
                </div>
            </div>
        </section>
        
        <section class="modules">
            <h2>Module Coverage</h2>
            <table>
                <thead>
                    <tr>
                        <th>Module</th>
                        <th>Coverage</th>
                        <th>Statements</th>
                        <th>Missing</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>"""

        for module_path, module_data in sorted(analysis['modules'].items()):
            status_class = module_data['status']
            html_content += f"""
                    <tr class="{status_class}">
                        <td>{module_path}</td>
                        <td>{module_data['coverage_percentage']:.2f}%</td>
                        <td>{module_data['statements']}</td>
                        <td>{module_data['missing']}</td>
                        <td class="status">{module_data['status'].upper()}</td>
                    </tr>"""

        html_content += """
                </tbody>
            </table>
        </section>
    </main>
</body>
</html>"""

        with open(output_dir / 'index.html', 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _generate_basic_css(self, output_dir: Path) -> None:
        """Generate basic CSS styles for HTML report."""
        css_content = """
/* FlyrigLoader Coverage Report Styles */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f5f5f5;
    color: #333;
}

header {
    background-color: #2E7D32;
    color: white;
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 20px;
}

header h1 {
    margin: 0;
    font-size: 2em;
}

.summary {
    background: white;
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.metrics {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
    margin-top: 15px;
}

.metric {
    display: flex;
    justify-content: space-between;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
}

.metric .label {
    font-weight: bold;
}

.metric .value {
    color: #2E7D32;
    font-weight: bold;
}

.modules {
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 15px;
}

th, td {
    padding: 10px;
    text-align: left;
    border-bottom: 1px solid #ddd;
}

th {
    background-color: #f8f9fa;
    font-weight: bold;
}

tr.excellent {
    background-color: #e8f5e8;
}

tr.good {
    background-color: #fff3cd;
}

tr.warning {
    background-color: #f8d7da;
}

tr.critical {
    background-color: #f5c6cb;
}

.status {
    font-weight: bold;
    text-transform: uppercase;
}
"""
        
        with open(output_dir / 'style.css', 'w', encoding='utf-8') as f:
            f.write(css_content)

    def generate_xml_report(self, analysis_results: Dict[str, Any]) -> bool:
        """
        Generate XML coverage report for CI/CD integration with comprehensive
        metadata and quality gate information.
        
        Args:
            analysis_results: Coverage analysis results from analyze_coverage_data()
            
        Returns:
            True if report generation succeeded, False otherwise
        """
        self.logger.info("Generating XML coverage report")
        
        try:
            xml_config = self.config.get('report_formats', {}).get('xml', {})
            output_file = xml_config.get('output_file', 'coverage.xml')
            output_path = self.output_dir / output_file

            # Prepare template context
            template_context = {
                'config': self.config,
                'thresholds': self.thresholds,
                'analysis': analysis_results,
                'generation_time': datetime.now(timezone.utc),
                'git_info': self._get_git_info(),
                'system_info': self._get_system_info()
            }

            # Generate XML using template if available
            if self.templates_dir.joinpath('coverage.xml.j2').exists():
                template = self.jinja_env.get_template('coverage.xml.j2')
                xml_content = template.render(**template_context)
            else:
                self.logger.warning("coverage.xml.j2 template not found, generating basic XML report")
                xml_content = self._generate_basic_xml_report(template_context)

            # Write XML report
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(xml_content)

            self.generation_stats['reports_generated'].append({
                'format': 'XML',
                'output_path': str(output_path),
                'files_generated': 1
            })

            self.logger.info(f"Generated XML report: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to generate XML report: {e}")
            self.generation_stats['errors'].append(f"XML generation failed: {e}")
            return False

    def _generate_basic_xml_report(self, context: Dict[str, Any]) -> str:
        """Generate a basic XML report when templates are not available."""
        analysis = context['analysis']
        git_info = context['git_info']
        system_info = context['system_info']
        
        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<coverage version="{system_info["coverage_version"]}" timestamp="{context["generation_time"].isoformat()}" line-rate="{analysis["summary"]["overall_coverage_percentage"]/100:.4f}" branch-rate="{analysis["summary"]["branch_coverage_percentage"]/100:.4f}" lines-covered="{analysis["summary"]["covered_statements"]}" lines-valid="{analysis["summary"]["total_statements"]}" branches-covered="{analysis["summary"]["covered_branches"]}" branches-valid="{analysis["summary"]["total_branches"]}" complexity="0">',
            '  <!-- FlyrigLoader Coverage Report -->',
            '  <sources>',
            f'    <source>{self.project_root}/src</source>',
            '  </sources>',
            '  <packages>'
        ]

        # Group modules by package
        packages = {}
        for module_path, module_data in analysis['modules'].items():
            package_name = module_path.split('/')[1] if '/' in module_path else 'flyrigloader'
            if package_name not in packages:
                packages[package_name] = []
            packages[package_name].append((module_path, module_data))

        # Generate package and class entries
        for package_name, modules in packages.items():
            package_statements = sum(m[1]['statements'] for m in modules)
            package_covered = sum(m[1]['covered'] for m in modules)
            package_rate = (package_covered / package_statements) if package_statements > 0 else 0

            xml_lines.append(f'    <package name="{package_name}" line-rate="{package_rate:.4f}" branch-rate="{package_rate:.4f}">')
            xml_lines.append('      <classes>')

            for module_path, module_data in modules:
                class_name = module_path.replace('/', '.').replace('.py', '')
                line_rate = module_data['coverage_percentage'] / 100
                xml_lines.append(f'        <class name="{class_name}" filename="{module_path}" line-rate="{line_rate:.4f}" branch-rate="{line_rate:.4f}">')
                xml_lines.append('          <methods/>')
                xml_lines.append('          <lines>')

                # Add line information
                for line_num in range(1, module_data['statements'] + module_data['missing'] + 1):
                    hits = 1 if line_num not in module_data['missing_lines'] else 0
                    xml_lines.append(f'            <line number="{line_num}" hits="{hits}"/>')

                xml_lines.append('          </lines>')
                xml_lines.append('        </class>')

            xml_lines.append('      </classes>')
            xml_lines.append('    </package>')

        xml_lines.extend([
            '  </packages>',
            '</coverage>'
        ])

        return '\n'.join(xml_lines)

    def generate_json_report(self, analysis_results: Dict[str, Any]) -> bool:
        """
        Generate JSON coverage report for programmatic analysis with comprehensive
        metrics, trend data, and quality gate information.
        
        Args:
            analysis_results: Coverage analysis results from analyze_coverage_data()
            
        Returns:
            True if report generation succeeded, False otherwise
        """
        self.logger.info("Generating JSON coverage report")
        
        try:
            json_config = self.config.get('report_formats', {}).get('json', {})
            output_file = json_config.get('output_file', 'coverage.json')
            output_path = self.output_dir / output_file

            # Prepare comprehensive JSON structure
            json_data = {
                'meta': {
                    'version': '2.0',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'generator': 'flyrigloader-coverage-generator',
                    'format_version': json_config.get('format_version', '2.0')
                },
                'project': {
                    'name': self.config.get('metadata', {}).get('project', 'flyrigloader'),
                    'version': 'latest',
                    'git_info': self._get_git_info(),
                    'system_info': self._get_system_info()
                },
                'coverage': {
                    'summary': analysis_results['summary'],
                    'modules': analysis_results['modules'],
                    'quality_gates': analysis_results['quality_gates'],
                    'statistics': analysis_results['statistics'],
                    'critical_modules': analysis_results['critical_modules']
                },
                'configuration': {
                    'thresholds': self.thresholds,
                    'report_config': self.config
                },
                'trend_analysis': self._calculate_coverage_trend(
                    analysis_results['summary']['overall_coverage_percentage']
                ),
                'generation_stats': self.generation_stats
            }

            # Use template if available
            if self.templates_dir.joinpath('coverage.json.j2').exists():
                template_context = {
                    'json_data': json_data,
                    'config': self.config,
                    'analysis': analysis_results
                }
                template = self.jinja_env.get_template('coverage.json.j2')
                json_content = template.render(**template_context)
                
                # Parse and re-serialize to ensure valid JSON
                try:
                    parsed_json = json.loads(json_content)
                    json_content = json.dumps(parsed_json, indent=2, ensure_ascii=False)
                except json.JSONDecodeError:
                    # Fallback to direct serialization
                    json_content = json.dumps(json_data, indent=2, ensure_ascii=False)
            else:
                self.logger.warning("coverage.json.j2 template not found, generating basic JSON report")
                json_content = json.dumps(json_data, indent=2, ensure_ascii=False)

            # Write JSON report
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_content)

            self.generation_stats['reports_generated'].append({
                'format': 'JSON',
                'output_path': str(output_path),
                'files_generated': 1
            })

            self.logger.info(f"Generated JSON report: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to generate JSON report: {e}")
            self.generation_stats['errors'].append(f"JSON generation failed: {e}")
            return False

    def generate_terminal_report(self, analysis_results: Dict[str, Any]) -> bool:
        """
        Generate terminal coverage report with colored output and summary statistics.
        
        Args:
            analysis_results: Coverage analysis results from analyze_coverage_data()
            
        Returns:
            True if report generation succeeded, False otherwise
        """
        self.logger.info("Generating terminal coverage report")
        
        try:
            terminal_config = self.config.get('report_formats', {}).get('terminal', {})
            
            # ANSI color codes
            colors = {
                'green': '\033[92m',
                'yellow': '\033[93m',
                'red': '\033[91m',
                'cyan': '\033[96m',
                'bold': '\033[1m',
                'reset': '\033[0m'
            }

            # Print header
            print(f"\n{colors['bold']}{colors['cyan']}FlyrigLoader Coverage Report{colors['reset']}")
            print(f"{colors['cyan']}{'=' * 50}{colors['reset']}")
            print(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
            
            # Print summary
            summary = analysis_results['summary']
            overall_pct = summary['overall_coverage_percentage']
            branch_pct = summary['branch_coverage_percentage']
            
            overall_color = colors['green'] if overall_pct >= 90 else colors['yellow'] if overall_pct >= 75 else colors['red']
            branch_color = colors['green'] if branch_pct >= 90 else colors['yellow'] if branch_pct >= 75 else colors['red']
            
            print(f"\n{colors['bold']}Summary Statistics:{colors['reset']}")
            print(f"  Total Files: {summary['total_files']}")
            print(f"  Total Statements: {summary['total_statements']:,}")
            print(f"  Covered Statements: {summary['covered_statements']:,}")
            print(f"  Overall Coverage: {overall_color}{overall_pct:.2f}%{colors['reset']}")
            print(f"  Branch Coverage: {branch_color}{branch_pct:.2f}%{colors['reset']}")

            # Print quality gates status
            quality_gates = analysis_results['quality_gates']
            print(f"\n{colors['bold']}Quality Gates:{colors['reset']}")
            
            for gate_name, gate_info in quality_gates.items():
                if isinstance(gate_info, dict) and 'status' in gate_info:
                    status_color = colors['green'] if gate_info['status'] == 'PASS' else colors['red']
                    print(f"  {gate_name}: {status_color}{gate_info['status']}{colors['reset']}")

            # Print module breakdown if requested
            if terminal_config.get('module_breakdown', True):
                print(f"\n{colors['bold']}Module Coverage:{colors['reset']}")
                print(f"{'Module':<50} {'Coverage':<10} {'Status':<10}")
                print(f"{'-' * 70}")
                
                for module_path, module_data in sorted(analysis_results['modules'].items()):
                    coverage_pct = module_data['coverage_percentage']
                    status = module_data['status']
                    
                    if status == 'excellent':
                        status_color = colors['green']
                    elif status == 'good':
                        status_color = colors['yellow']
                    else:
                        status_color = colors['red']
                    
                    print(f"{module_path:<50} {coverage_pct:>7.2f}% {status_color}{status:<10}{colors['reset']}")

            # Print critical modules status
            critical_modules = analysis_results['critical_modules']
            if critical_modules['failing_critical'] > 0:
                print(f"\n{colors['bold']}{colors['red']}Critical Module Failures:{colors['reset']}")
                for category, modules in critical_modules.items():
                    if isinstance(modules, list):
                        for module_info in modules:
                            if module_info['status'] == 'FAIL':
                                print(f"  {colors['red']}{module_info['module']}: {module_info['coverage']:.2f}%{colors['reset']}")

            print()  # Final newline
            return True

        except Exception as e:
            self.logger.error(f"Failed to generate terminal report: {e}")
            self.generation_stats['errors'].append(f"Terminal generation failed: {e}")
            return False

    def run_full_pipeline(self) -> int:
        """
        Execute the complete coverage report generation pipeline including
        data loading, analysis, and multi-format report generation.
        
        Returns:
            Exit code: 0 for success, 1 for failure
        """
        self.logger.info("Starting complete coverage report generation pipeline")
        
        try:
            # Load configuration and data
            self.load_configuration()
            self.load_coverage_data()
            self.setup_jinja_environment()
            
            # Perform coverage analysis
            analysis_results = self.analyze_coverage_data()
            
            # Track report generation success
            reports_enabled = self.config.get('report_formats', {})
            successful_reports = 0
            total_reports = 0
            
            # Generate enabled reports
            if reports_enabled.get('html', {}).get('enabled', True):
                total_reports += 1
                if self.generate_html_report(analysis_results):
                    successful_reports += 1
            
            if reports_enabled.get('xml', {}).get('enabled', True):
                total_reports += 1
                if self.generate_xml_report(analysis_results):
                    successful_reports += 1
            
            if reports_enabled.get('json', {}).get('enabled', True):
                total_reports += 1
                if self.generate_json_report(analysis_results):
                    successful_reports += 1
            
            if reports_enabled.get('terminal', {}).get('enabled', True):
                total_reports += 1
                if self.generate_terminal_report(analysis_results):
                    successful_reports += 1

            # Update generation statistics
            self.generation_stats['end_time'] = datetime.now(timezone.utc)
            self.generation_stats['duration'] = time.time() - self.start_time
            self.generation_stats['reports_successful'] = successful_reports
            self.generation_stats['reports_total'] = total_reports

            # Log summary
            self.logger.info(f"Coverage report generation completed")
            self.logger.info(f"Reports generated: {successful_reports}/{total_reports}")
            self.logger.info(f"Overall coverage: {analysis_results['summary']['overall_coverage_percentage']:.2f}%")
            self.logger.info(f"Generation time: {self.generation_stats['duration']:.2f} seconds")

            # Check quality gates for exit code
            quality_gates = analysis_results['quality_gates']
            all_gates_passed = all(
                gate_info.get('passed', False) for gate_info in quality_gates.values()
                if isinstance(gate_info, dict) and 'passed' in gate_info
            )

            if not all_gates_passed and self.thresholds.get('quality_gates', {}).get('enforcement_rules', {}).get('fail_ci_on_violation', True):
                self.logger.error("Quality gates failed - returning exit code 1")
                return 1

            if successful_reports < total_reports:
                self.logger.warning(f"Some reports failed to generate ({successful_reports}/{total_reports})")
                return 1

            return 0

        except Exception as e:
            self.logger.error(f"Coverage report generation pipeline failed: {e}")
            return 1


def main() -> int:
    """
    Main entry point for the coverage report generation script.
    
    Supports command-line arguments for configuration and provides
    comprehensive error handling and logging.
    
    Returns:
        Exit code: 0 for success, 1 for failure
    """
    parser = argparse.ArgumentParser(
        description="FlyrigLoader Coverage Report Generator - Comprehensive multi-format coverage reporting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Generate all reports with default settings
  %(prog)s --verbose                          # Enable verbose logging
  %(prog)s --coverage-data .coverage          # Specify coverage data file
  %(prog)s --output-dir ./reports             # Specify output directory
  %(prog)s --config ./custom-config.json     # Use custom configuration
  %(prog)s --thresholds ./custom-thresholds.json  # Use custom thresholds

Report Formats:
  - HTML: Interactive coverage report with source code highlighting
  - XML: Machine-readable format for CI/CD integration
  - JSON: Programmatic analysis with comprehensive metrics
  - Terminal: Colored console output with summary statistics

Quality Gates:
  The script validates coverage against configurable thresholds and returns
  appropriate exit codes for CI/CD integration:
  - Exit code 0: All quality gates passed
  - Exit code 1: Quality gate failures or generation errors
        """
    )
    
    parser.add_argument(
        '--coverage-data', 
        help='Path to coverage data file (default: .coverage)',
        default=None
    )
    
    parser.add_argument(
        '--config', 
        help='Path to report configuration JSON file (default: tests/coverage/report-config.json)',
        default=None
    )
    
    parser.add_argument(
        '--thresholds',
        help='Path to coverage thresholds JSON file (default: tests/coverage/coverage-thresholds.json)',
        default=None
    )
    
    parser.add_argument(
        '--output-dir',
        help='Output directory for generated reports (default: htmlcov)',
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
        version='FlyrigLoader Coverage Report Generator 1.0.0'
    )

    args = parser.parse_args()

    # Initialize and run the coverage report generator
    try:
        generator = CoverageReportGenerator(
            coverage_data_file=args.coverage_data,
            config_file=args.config,
            thresholds_file=args.thresholds,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
        
        return generator.run_full_pipeline()
        
    except KeyboardInterrupt:
        print("\nCoverage report generation interrupted by user")
        return 1
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())