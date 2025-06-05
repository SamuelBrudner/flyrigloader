#!/bin/bash

# =============================================================================
# CI/CD Environment Preparation Script for FlyRigLoader
# =============================================================================
#
# Comprehensive CI/CD environment setup implementing automated dependency 
# installation, test environment configuration, and quality assurance tool 
# setup per Section 3.6.1 and 3.6.4 requirements.
#
# Features:
# - SEC-3.6.1: Development environment management with comprehensive testing infrastructure dependencies
# - SEC-3.6.3: Testing infrastructure with PyTest framework and advanced coverage integration  
# - SEC-3.6.4: Quality assurance with automated quality gates enforcement
# - SEC-4.1.1.5: Test execution workflow with environment setup requirements
#
# Performance Requirements:
# - Environment setup must complete within 5 minutes in CI/CD pipelines
# - Dependency installation with parallel processing where supported
# - Automated validation of tool versions and configurations
# - Comprehensive error handling and recovery mechanisms
#
# Usage:
#   ./tests/coverage/scripts/prepare-ci-environment.sh [OPTIONS]
#
# Options:
#   --conda           Use conda for dependency management (default if environment.yml exists)
#   --pip             Use pip for dependency management  
#   --skip-conda      Skip conda environment setup
#   --skip-precommit  Skip pre-commit hook installation
#   --validate-only   Only validate existing environment setup
#   --verbose         Enable verbose output for debugging
#   --help            Show this help message
#
# Environment Variables:
#   CI                   Set to 'true' in CI/CD environments
#   PYTHON_VERSION       Target Python version (default: auto-detect)
#   COVERAGE_THRESHOLD   Minimum coverage threshold (default: 90)
#   PYTEST_WORKERS      Number of parallel pytest workers (default: auto)
#
# =============================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# =============================================================================
# Configuration and Constants
# =============================================================================

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
readonly LOG_FILE="${PROJECT_ROOT}/tests/coverage/environment_setup.log"

# Performance and quality thresholds per technical specification
readonly MIN_PYTHON_VERSION="3.8"
readonly MAX_PYTHON_VERSION="3.11" 
readonly DEFAULT_COVERAGE_THRESHOLD="90"
readonly SETUP_TIMEOUT="300"  # 5 minutes maximum setup time
readonly RETRY_ATTEMPTS="3"

# Tool version requirements per Section 3.6.3 and 3.6.4
readonly PYTEST_MIN_VERSION="7.0.0"
readonly PYTEST_COV_MIN_VERSION="6.1.1"
readonly BLACK_MIN_VERSION="24.3.0"
readonly MYPY_MIN_VERSION="1.8.0"
readonly COVERAGE_MIN_VERSION="7.8.2"

# Color codes for output formatting
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# =============================================================================
# Global Variables
# =============================================================================

USE_CONDA=false
USE_PIP=false
SKIP_CONDA=false
SKIP_PRECOMMIT=false
VALIDATE_ONLY=false
VERBOSE=false
PYTHON_CMD="python"
PIP_CMD="pip"
CONDA_CMD=""

# =============================================================================
# Utility Functions
# =============================================================================

# Logging function with timestamp and color support
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    
    case "$level" in
        "INFO")  echo -e "${GREEN}[INFO]${NC}  ${timestamp} - $message" | tee -a "$LOG_FILE" ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC}  ${timestamp} - $message" | tee -a "$LOG_FILE" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE" ;;
        "DEBUG") [[ "$VERBOSE" == true ]] && echo -e "${BLUE}[DEBUG]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE" ;;
        "SUCCESS") echo -e "${GREEN}[SUCCESS]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE" ;;
        *)       echo -e "${CYAN}[$level]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE" ;;
    esac
}

# Enhanced error handling with context and recovery suggestions
handle_error() {
    local exit_code=$?
    local line_number="$1"
    local command="$2"
    
    log "ERROR" "Command failed at line $line_number: $command (exit code: $exit_code)"
    log "ERROR" "Environment setup failed. Check $LOG_FILE for detailed information."
    log "INFO" "Common solutions:"
    log "INFO" "  1. Ensure Python ${MIN_PYTHON_VERSION}+ is installed and accessible"
    log "INFO" "  2. Check network connectivity for package downloads"
    log "INFO" "  3. Verify sufficient disk space (>2GB recommended)"
    log "INFO" "  4. Run with --verbose flag for detailed debugging output"
    
    # Cleanup on failure
    cleanup_on_failure
    exit $exit_code
}

# Cleanup function for failed setups
cleanup_on_failure() {
    log "INFO" "Performing cleanup after failed setup..."
    
    # Remove incomplete installations
    if [[ -f "${PROJECT_ROOT}/.pytest_cache" ]]; then
        rm -rf "${PROJECT_ROOT}/.pytest_cache"
        log "DEBUG" "Removed incomplete pytest cache"
    fi
    
    # Reset git hooks if partially installed
    if [[ -d "${PROJECT_ROOT}/.git/hooks" ]] && [[ -f "${PROJECT_ROOT}/.git/hooks/pre-commit" ]]; then
        git config --unset-all core.hooksPath 2>/dev/null || true
        log "DEBUG" "Reset git hooks configuration"
    fi
}

# Trap errors with context information
trap 'handle_error ${LINENO} "$BASH_COMMAND"' ERR

# Version comparison utility
version_compare() {
    local version1="$1"
    local version2="$2"
    
    # Convert versions to comparable integers
    local ver1=$(echo "$version1" | sed 's/\.//g' | sed 's/[^0-9]//g')
    local ver2=$(echo "$version2" | sed 's/\.//g' | sed 's/[^0-9]//g')
    
    # Pad shorter version with zeros
    while [[ ${#ver1} -lt ${#ver2} ]]; do ver1="${ver1}0"; done
    while [[ ${#ver2} -lt ${#ver1} ]]; do ver2="${ver2}0"; done
    
    [[ "$ver1" -ge "$ver2" ]]
}

# Network connectivity check
check_network_connectivity() {
    log "DEBUG" "Checking network connectivity for package downloads..."
    
    local test_urls=(
        "https://pypi.org"
        "https://conda.anaconda.org"
        "https://github.com"
    )
    
    for url in "${test_urls[@]}"; do
        if ! curl -s --head --max-time 10 "$url" > /dev/null 2>&1; then
            log "WARN" "Cannot reach $url - package installation may fail"
        else
            log "DEBUG" "Successfully connected to $url"
        fi
    done
}

# =============================================================================
# Environment Detection and Validation
# =============================================================================

# Detect and validate Python installation
detect_python_environment() {
    log "INFO" "Detecting Python environment configuration..."
    
    # Find Python executable
    for python_candidate in python3 python python3.11 python3.10 python3.9 python3.8; do
        if command -v "$python_candidate" >/dev/null 2>&1; then
            PYTHON_CMD="$python_candidate"
            break
        fi
    done
    
    if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
        log "ERROR" "No suitable Python installation found"
        log "ERROR" "Please install Python ${MIN_PYTHON_VERSION} or higher"
        exit 1
    fi
    
    # Validate Python version
    local python_version
    python_version=$("$PYTHON_CMD" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    
    log "INFO" "Found Python $python_version at $(which "$PYTHON_CMD")"
    
    if ! version_compare "$python_version" "$MIN_PYTHON_VERSION"; then
        log "ERROR" "Python version $python_version is below minimum requirement $MIN_PYTHON_VERSION"
        exit 1
    fi
    
    if ! version_compare "$MAX_PYTHON_VERSION" "$python_version"; then
        log "WARN" "Python version $python_version is above tested maximum $MAX_PYTHON_VERSION"
        log "WARN" "Proceeding with setup but compatibility is not guaranteed"
    fi
    
    # Set pip command based on Python executable
    PIP_CMD="${PYTHON_CMD} -m pip"
    
    log "SUCCESS" "Python environment validation completed"
}

# Detect conda installation and environment
detect_conda_environment() {
    log "INFO" "Detecting conda environment configuration..."
    
    # Find conda executable
    for conda_candidate in conda mamba micromamba; do
        if command -v "$conda_candidate" >/dev/null 2>&1; then
            CONDA_CMD="$conda_candidate"
            break
        fi
    done
    
    if [[ -n "$CONDA_CMD" ]]; then
        local conda_version
        conda_version=$("$CONDA_CMD" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
        log "INFO" "Found $CONDA_CMD version $conda_version"
        
        # Check if we're in a conda environment
        if [[ -n "${CONDA_DEFAULT_ENV:-}" ]] && [[ "$CONDA_DEFAULT_ENV" != "base" ]]; then
            log "INFO" "Current conda environment: $CONDA_DEFAULT_ENV"
        fi
        
        return 0
    else
        log "WARN" "No conda installation detected"
        return 1
    fi
}

# Validate project structure and required files
validate_project_structure() {
    log "INFO" "Validating project structure and configuration files..."
    
    local required_files=(
        "pyproject.toml"
        "src/flyrigloader/__init__.py"
        "tests"
    )
    
    local optional_files=(
        "environment.yml"
        "tests/coverage/pytest.ini"
        ".pre-commit-config.yaml"
    )
    
    # Check required files
    for file in "${required_files[@]}"; do
        local filepath="${PROJECT_ROOT}/$file"
        if [[ ! -e "$filepath" ]]; then
            log "ERROR" "Required file/directory not found: $file"
            exit 1
        fi
        log "DEBUG" "Found required file: $file"
    done
    
    # Check optional files and set configuration flags
    for file in "${optional_files[@]}"; do
        local filepath="${PROJECT_ROOT}/$file"
        if [[ -e "$filepath" ]]; then
            log "DEBUG" "Found optional file: $file"
            case "$file" in
                "environment.yml") USE_CONDA=true ;;
                ".pre-commit-config.yaml") SKIP_PRECOMMIT=false ;;
            esac
        else
            log "DEBUG" "Optional file not found: $file"
        fi
    done
    
    log "SUCCESS" "Project structure validation completed"
}

# =============================================================================
# Dependency Management
# =============================================================================

# Install dependencies using conda
install_conda_dependencies() {
    if [[ "$SKIP_CONDA" == true ]] || [[ -z "$CONDA_CMD" ]]; then
        log "INFO" "Skipping conda dependency installation"
        return 0
    fi
    
    log "INFO" "Installing dependencies using conda from environment.yml..."
    
    local env_file="${PROJECT_ROOT}/environment.yml"
    if [[ ! -f "$env_file" ]]; then
        log "ERROR" "environment.yml not found at $env_file"
        return 1
    fi
    
    # Check if environment already exists
    local env_name
    env_name=$(grep "^name:" "$env_file" | cut -d: -f2 | tr -d ' ')
    
    if [[ -n "$env_name" ]] && "$CONDA_CMD" env list | grep -q "^$env_name "; then
        log "INFO" "Conda environment '$env_name' already exists, updating..."
        timeout "$SETUP_TIMEOUT" "$CONDA_CMD" env update -f "$env_file" --prune
    else
        log "INFO" "Creating new conda environment '$env_name'..."
        timeout "$SETUP_TIMEOUT" "$CONDA_CMD" env create -f "$env_file"
    fi
    
    # Activate environment if not in CI
    if [[ "${CI:-false}" != "true" ]] && [[ -n "$env_name" ]]; then
        log "INFO" "To activate the environment, run: conda activate $env_name"
    fi
    
    log "SUCCESS" "Conda dependencies installed successfully"
}

# Install dependencies using pip
install_pip_dependencies() {
    log "INFO" "Installing Python dependencies using pip..."
    
    # Upgrade pip first
    log "DEBUG" "Upgrading pip to latest version..."
    "$PIP_CMD" install --upgrade pip setuptools wheel
    
    # Install project in development mode with all dependencies
    log "DEBUG" "Installing project with development dependencies..."
    if [[ -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
        timeout "$SETUP_TIMEOUT" "$PIP_CMD" install -e "${PROJECT_ROOT}[dev]"
    else
        log "ERROR" "pyproject.toml not found for pip installation"
        return 1
    fi
    
    log "SUCCESS" "Pip dependencies installed successfully"
}

# Validate installed package versions
validate_tool_versions() {
    log "INFO" "Validating installed tool versions against requirements..."
    
    local validation_errors=0
    
    # Define tools and their minimum versions
    declare -A tools=(
        ["pytest"]="$PYTEST_MIN_VERSION"
        ["pytest-cov"]="$PYTEST_COV_MIN_VERSION"
        ["black"]="$BLACK_MIN_VERSION"
        ["mypy"]="$MYPY_MIN_VERSION"
        ["coverage"]="$COVERAGE_MIN_VERSION"
    )
    
    for tool in "${!tools[@]}"; do
        local min_version="${tools[$tool]}"
        local installed_version
        
        case "$tool" in
            "pytest"|"black"|"mypy"|"coverage")
                if command -v "$tool" >/dev/null 2>&1; then
                    installed_version=$("$tool" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
                else
                    log "ERROR" "$tool is not installed or not in PATH"
                    ((validation_errors++))
                    continue
                fi
                ;;
            "pytest-cov")
                installed_version=$("$PYTHON_CMD" -c "import pytest_cov; print(pytest_cov.__version__)" 2>/dev/null || echo "not_found")
                if [[ "$installed_version" == "not_found" ]]; then
                    log "ERROR" "pytest-cov is not installed"
                    ((validation_errors++))
                    continue
                fi
                ;;
        esac
        
        if version_compare "$installed_version" "$min_version"; then
            log "SUCCESS" "$tool $installed_version (>= $min_version) ✓"
        else
            log "ERROR" "$tool $installed_version is below minimum requirement $min_version"
            ((validation_errors++))
        fi
    done
    
    if [[ $validation_errors -gt 0 ]]; then
        log "ERROR" "Tool version validation failed with $validation_errors errors"
        return 1
    fi
    
    log "SUCCESS" "All tool versions validated successfully"
}

# =============================================================================
# Testing Infrastructure Setup
# =============================================================================

# Configure pytest and coverage tools
setup_testing_infrastructure() {
    log "INFO" "Setting up testing infrastructure with pytest and coverage integration..."
    
    # Create necessary directories
    local test_dirs=(
        "tests/logs"
        "tests/coverage/reports"
        ".benchmarks"
        "htmlcov"
    )
    
    for dir in "${test_dirs[@]}"; do
        local full_path="${PROJECT_ROOT}/$dir"
        if [[ ! -d "$full_path" ]]; then
            mkdir -p "$full_path"
            log "DEBUG" "Created directory: $dir"
        fi
    done
    
    # Set up coverage configuration if not exists
    setup_coverage_configuration
    
    # Configure pytest cache and temporary directories
    export PYTEST_CACHE_DIR="${PROJECT_ROOT}/.pytest_cache"
    export PYTEST_TMP_DIR="${PROJECT_ROOT}/tests/tmp"
    
    # Set coverage context for CI environments
    if [[ "${CI:-false}" == "true" ]]; then
        export COVERAGE_CONTEXT="ci"
    else
        export COVERAGE_CONTEXT="local"
    fi
    
    # Configure parallel test execution
    local cpu_count
    cpu_count=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "2")
    export PYTEST_WORKERS="${PYTEST_WORKERS:-$cpu_count}"
    
    log "DEBUG" "Configured pytest with $PYTEST_WORKERS parallel workers"
    log "SUCCESS" "Testing infrastructure setup completed"
}

# Set up coverage configuration and reporting
setup_coverage_configuration() {
    log "INFO" "Configuring coverage reporting with custom templates and thresholds..."
    
    local coverage_threshold="${COVERAGE_THRESHOLD:-$DEFAULT_COVERAGE_THRESHOLD}"
    
    # Create .coveragerc if it doesn't exist
    local coveragerc="${PROJECT_ROOT}/.coveragerc"
    if [[ ! -f "$coveragerc" ]]; then
        log "DEBUG" "Creating .coveragerc configuration file..."
        cat > "$coveragerc" << EOF
# Coverage configuration for FlyRigLoader
# Generated by prepare-ci-environment.sh

[run]
source = src/flyrigloader
branch = true
parallel = true
context = \${COVERAGE_CONTEXT}
omit = 
    */tests/*
    */test_*
    */__pycache__/*
    */site-packages/*
    */.*

[report]
show_missing = true
fail_under = $coverage_threshold
precision = 2
exclude_lines =
    pragma: no cover
    def __repr__
    if self.debug:
    if settings.DEBUG
    raise AssertionError
    raise NotImplementedError
    if 0:
    if __name__ == .__main__.:

[html]
directory = htmlcov
title = FlyRigLoader Test Coverage Report

[xml]
output = coverage.xml

[json]
output = coverage.json
EOF
        log "DEBUG" "Created .coveragerc with $coverage_threshold% threshold"
    fi
    
    # Create coverage report template
    local template_dir="${PROJECT_ROOT}/tests/coverage/templates"
    mkdir -p "$template_dir"
    
    local coverage_template="${template_dir}/coverage_report.html"
    if [[ ! -f "$coverage_template" ]]; then
        log "DEBUG" "Creating custom coverage report template..."
        cat > "$coverage_template" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>FlyRigLoader Coverage Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 15px; border-radius: 5px; }
        .summary { margin: 20px 0; }
        .threshold { font-weight: bold; color: #007700; }
        .warning { color: #cc6600; }
        .error { color: #cc0000; }
    </style>
</head>
<body>
    <div class="header">
        <h1>FlyRigLoader Test Coverage Report</h1>
        <p>Generated on: {{ timestamp }}</p>
        <p>Coverage Threshold: <span class="threshold">{{ threshold }}%</span></p>
    </div>
    <div class="summary">
        {{ coverage_summary }}
    </div>
</body>
</html>
EOF
        log "DEBUG" "Created custom coverage report template"
    fi
    
    log "SUCCESS" "Coverage configuration setup completed"
}

# =============================================================================
# Quality Assurance Tools Setup
# =============================================================================

# Configure code quality tools (black, isort, mypy, flake8)
setup_quality_assurance_tools() {
    log "INFO" "Setting up quality assurance tools with automated enforcement..."
    
    # Configure black for code formatting
    setup_black_configuration
    
    # Configure isort for import sorting
    setup_isort_configuration
    
    # Configure mypy for type checking
    setup_mypy_configuration
    
    # Configure flake8 for linting
    setup_flake8_configuration
    
    log "SUCCESS" "Quality assurance tools setup completed"
}

# Configure black code formatter
setup_black_configuration() {
    log "DEBUG" "Configuring black code formatter..."
    
    local pyproject_toml="${PROJECT_ROOT}/pyproject.toml"
    
    # Check if black configuration exists in pyproject.toml
    if ! grep -q "\[tool\.black\]" "$pyproject_toml" 2>/dev/null; then
        log "DEBUG" "Adding black configuration to pyproject.toml..."
        cat >> "$pyproject_toml" << 'EOF'

[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310', 'py311']
include = '\.pyi?$'
exclude = '''
/(
    \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | buck-out
  | build
  | dist
  | __pycache__
)/
'''
EOF
    fi
}

# Configure isort import sorting
setup_isort_configuration() {
    log "DEBUG" "Configuring isort import sorting..."
    
    local pyproject_toml="${PROJECT_ROOT}/pyproject.toml"
    
    # Check if isort configuration exists
    if ! grep -q "\[tool\.isort\]" "$pyproject_toml" 2>/dev/null; then
        log "DEBUG" "Adding isort configuration to pyproject.toml..."
        cat >> "$pyproject_toml" << 'EOF'

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["flyrigloader"]
known_third_party = ["numpy", "pandas", "pydantic", "loguru", "pytest"]
sections = ["FUTURE", "STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"]
EOF
    fi
}

# Configure mypy type checking
setup_mypy_configuration() {
    log "DEBUG" "Configuring mypy type checking..."
    
    local mypy_ini="${PROJECT_ROOT}/mypy.ini"
    if [[ ! -f "$mypy_ini" ]]; then
        log "DEBUG" "Creating mypy.ini configuration file..."
        cat > "$mypy_ini" << 'EOF'
[mypy]
python_version = 3.8
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
strict_equality = True
show_error_codes = True

[mypy-tests.*]
disallow_untyped_defs = False
disallow_incomplete_defs = False

[mypy-numpy.*]
ignore_missing_imports = True

[mypy-pandas.*]
ignore_missing_imports = True

[mypy-pytest.*]
ignore_missing_imports = True
EOF
    fi
}

# Configure flake8 linting
setup_flake8_configuration() {
    log "DEBUG" "Configuring flake8 linting..."
    
    local flake8_config="${PROJECT_ROOT}/.flake8"
    if [[ ! -f "$flake8_config" ]]; then
        log "DEBUG" "Creating .flake8 configuration file..."
        cat > "$flake8_config" << 'EOF'
[flake8]
max-line-length = 88
extend-ignore = E203, E266, E501, W503, F403, F401
max-complexity = 10
select = B,C,E,F,W,T4,B9
exclude = 
    .git,
    __pycache__,
    .pytest_cache,
    .mypy_cache,
    build,
    dist,
    *.egg-info,
    .venv,
    .env
per-file-ignores =
    __init__.py:F401
    tests/*:F401,F811,F841
EOF
    fi
}

# =============================================================================
# Pre-commit Hooks Setup
# =============================================================================

# Install and configure pre-commit hooks
setup_precommit_hooks() {
    if [[ "$SKIP_PRECOMMIT" == true ]]; then
        log "INFO" "Skipping pre-commit hook installation"
        return 0
    fi
    
    log "INFO" "Setting up pre-commit hooks for automated quality enforcement..."
    
    # Create .pre-commit-config.yaml if it doesn't exist
    create_precommit_configuration
    
    # Install pre-commit hooks
    if command -v pre-commit >/dev/null 2>&1; then
        log "DEBUG" "Installing pre-commit hooks..."
        pre-commit install
        pre-commit install --hook-type commit-msg
        
        # Optionally run on all files for first-time setup
        if [[ "${CI:-false}" != "true" ]]; then
            log "INFO" "Running pre-commit on all files for initial validation..."
            pre-commit run --all-files || log "WARN" "Pre-commit found issues - please fix before committing"
        fi
        
        log "SUCCESS" "Pre-commit hooks installed successfully"
    else
        log "ERROR" "pre-commit command not found - ensure it's installed"
        return 1
    fi
}

# Create pre-commit configuration
create_precommit_configuration() {
    local precommit_config="${PROJECT_ROOT}/.pre-commit-config.yaml"
    
    if [[ -f "$precommit_config" ]]; then
        log "DEBUG" "Pre-commit configuration already exists"
        return 0
    fi
    
    log "DEBUG" "Creating .pre-commit-config.yaml..."
    cat > "$precommit_config" << 'EOF'
# Pre-commit hooks configuration for FlyRigLoader
# Generated by prepare-ci-environment.sh

repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/psf/black
    rev: 24.3.0
    hooks:
      - id: black
        language_version: python3
        args: [--line-length=88]

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: [--profile, black, --line-length=88]

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        additional_dependencies: [flake8-docstrings, flake8-bugbear]
        args: [--max-line-length=88, --extend-ignore=E203,W503]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--ignore-missing-imports]

  - repo: local
    hooks:
      - id: pytest-coverage
        name: pytest with coverage validation
        entry: pytest
        language: system
        args: [--cov=src/flyrigloader, --cov-fail-under=90, --cov-report=term-missing, --maxfail=5]
        types: [python]
        pass_filenames: false
        always_run: false
EOF
    
    log "DEBUG" "Created pre-commit configuration with quality gates"
}

# =============================================================================
# Environment Validation
# =============================================================================

# Comprehensive environment validation
validate_complete_environment() {
    log "INFO" "Performing comprehensive environment validation..."
    
    local validation_errors=0
    
    # Test pytest functionality
    log "DEBUG" "Testing pytest functionality..."
    if ! "$PYTHON_CMD" -m pytest --version >/dev/null 2>&1; then
        log "ERROR" "pytest is not working correctly"
        ((validation_errors++))
    fi
    
    # Test coverage functionality
    log "DEBUG" "Testing coverage functionality..."
    if ! "$PYTHON_CMD" -m coverage --version >/dev/null 2>&1; then
        log "ERROR" "coverage is not working correctly"
        ((validation_errors++))
    fi
    
    # Test code quality tools
    local quality_tools=("black" "isort" "mypy" "flake8")
    for tool in "${quality_tools[@]}"; do
        log "DEBUG" "Testing $tool functionality..."
        if ! command -v "$tool" >/dev/null 2>&1; then
            log "ERROR" "$tool is not available in PATH"
            ((validation_errors++))
        fi
    done
    
    # Test pre-commit if not skipped
    if [[ "$SKIP_PRECOMMIT" != true ]]; then
        log "DEBUG" "Testing pre-commit functionality..."
        if ! command -v pre-commit >/dev/null 2>&1; then
            log "ERROR" "pre-commit is not available"
            ((validation_errors++))
        fi
    fi
    
    # Run a minimal test to ensure the environment works
    log "DEBUG" "Running minimal test suite validation..."
    local test_output
    if ! test_output=$("$PYTHON_CMD" -c "
import sys
import pytest
import coverage
import numpy
import pandas
import pydantic
import loguru
print('All critical imports successful')
" 2>&1); then
        log "ERROR" "Critical import test failed: $test_output"
        ((validation_errors++))
    fi
    
    if [[ $validation_errors -eq 0 ]]; then
        log "SUCCESS" "Environment validation completed successfully"
        return 0
    else
        log "ERROR" "Environment validation failed with $validation_errors errors"
        return 1
    fi
}

# =============================================================================
# Main Execution Functions
# =============================================================================

# Display help information
show_help() {
    cat << 'EOF'
CI/CD Environment Preparation Script for FlyRigLoader

USAGE:
    ./tests/coverage/scripts/prepare-ci-environment.sh [OPTIONS]

DESCRIPTION:
    Comprehensive CI/CD environment setup implementing automated dependency 
    installation, test environment configuration, and quality assurance tool 
    setup per Section 3.6.1 and 3.6.4 requirements.

OPTIONS:
    --conda           Use conda for dependency management (default if environment.yml exists)
    --pip             Use pip for dependency management  
    --skip-conda      Skip conda environment setup
    --skip-precommit  Skip pre-commit hook installation
    --validate-only   Only validate existing environment setup
    --verbose         Enable verbose output for debugging
    --help            Show this help message

ENVIRONMENT VARIABLES:
    CI                   Set to 'true' in CI/CD environments
    PYTHON_VERSION       Target Python version (default: auto-detect)
    COVERAGE_THRESHOLD   Minimum coverage threshold (default: 90)
    PYTEST_WORKERS      Number of parallel pytest workers (default: auto)

EXAMPLES:
    # Standard setup with conda (recommended)
    ./tests/coverage/scripts/prepare-ci-environment.sh --conda

    # Setup with pip only
    ./tests/coverage/scripts/prepare-ci-environment.sh --pip --skip-conda

    # Validate existing environment
    ./tests/coverage/scripts/prepare-ci-environment.sh --validate-only

    # CI/CD environment setup
    CI=true ./tests/coverage/scripts/prepare-ci-environment.sh --verbose

REQUIREMENTS:
    - Python 3.8+ 
    - Internet connectivity for package downloads
    - Sufficient disk space (>2GB recommended)
    - Git repository (for pre-commit hooks)

For more information, see the technical specification sections:
    - Section 3.6.1: Development Environment Management
    - Section 3.6.3: Testing Infrastructure  
    - Section 3.6.4: Quality Assurance
    - Section 4.1.1.5: Test Execution Workflow
EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --conda)
                USE_CONDA=true
                shift
                ;;
            --pip)
                USE_PIP=true
                shift
                ;;
            --skip-conda)
                SKIP_CONDA=true
                shift
                ;;
            --skip-precommit)
                SKIP_PRECOMMIT=true
                shift
                ;;
            --validate-only)
                VALIDATE_ONLY=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log "ERROR" "Unknown option: $1"
                log "INFO" "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

# Main setup orchestration function
main() {
    local start_time
    start_time=$(date +%s)
    
    log "INFO" "Starting FlyRigLoader CI/CD environment preparation..."
    log "INFO" "Project root: $PROJECT_ROOT"
    log "INFO" "Log file: $LOG_FILE"
    
    # Initialize log file
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "# FlyRigLoader Environment Setup Log - $(date)" > "$LOG_FILE"
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Early validation mode
    if [[ "$VALIDATE_ONLY" == true ]]; then
        log "INFO" "Running validation-only mode..."
        detect_python_environment
        validate_tool_versions
        validate_complete_environment
        log "SUCCESS" "Environment validation completed"
        exit 0
    fi
    
    # Check network connectivity
    check_network_connectivity
    
    # Phase 1: Environment Detection
    log "INFO" "Phase 1: Environment Detection and Validation"
    detect_python_environment
    detect_conda_environment
    validate_project_structure
    
    # Phase 2: Dependency Installation
    log "INFO" "Phase 2: Dependency Installation"
    
    # Determine installation method
    if [[ "$USE_CONDA" == true ]] || [[ -f "${PROJECT_ROOT}/environment.yml" && "$USE_PIP" != true ]]; then
        install_conda_dependencies
    fi
    
    if [[ "$USE_PIP" == true ]] || [[ "$USE_CONDA" != true ]]; then
        install_pip_dependencies
    fi
    
    # Phase 3: Tool Configuration
    log "INFO" "Phase 3: Testing Infrastructure and Quality Assurance Setup"
    setup_testing_infrastructure
    setup_quality_assurance_tools
    
    # Phase 4: Pre-commit Setup
    log "INFO" "Phase 4: Pre-commit Hooks Installation"
    setup_precommit_hooks
    
    # Phase 5: Validation
    log "INFO" "Phase 5: Environment Validation"
    validate_tool_versions
    validate_complete_environment
    
    # Calculate setup time
    local end_time
    end_time=$(date +%s)
    local setup_duration=$((end_time - start_time))
    
    # Final success message
    log "SUCCESS" "Environment setup completed successfully in ${setup_duration} seconds"
    log "INFO" "Environment is ready for development and testing"
    
    # Display next steps
    cat << EOF

${GREEN}✓ Environment Setup Complete${NC}

${CYAN}Next Steps:${NC}
  1. Activate your environment:
     ${YELLOW}conda activate flyrigloader-dev${NC}  (if using conda)
     
  2. Run the test suite:
     ${YELLOW}pytest --cov=src/flyrigloader --cov-report=html${NC}
     
  3. Check code quality:
     ${YELLOW}pre-commit run --all-files${NC}
     
  4. View coverage report:
     ${YELLOW}open htmlcov/index.html${NC}

${CYAN}Environment Summary:${NC}
  • Python: $("$PYTHON_CMD" --version)
  • Test framework: pytest with coverage and benchmarking
  • Code quality: black, isort, mypy, flake8  
  • Pre-commit hooks: $(if [[ "$SKIP_PRECOMMIT" != true ]]; then echo "✓ Installed"; else echo "✗ Skipped"; fi)
  • Coverage threshold: ${COVERAGE_THRESHOLD:-$DEFAULT_COVERAGE_THRESHOLD}%

For detailed logs, see: ${LOG_FILE}

EOF
}

# =============================================================================
# Script Execution
# =============================================================================

# Change to project root directory
cd "$PROJECT_ROOT"

# Execute main function with all arguments
main "$@"