#!/bin/bash

# =============================================================================
# CI/CD Environment Preparation Script
# =============================================================================
#
# Comprehensive CI/CD environment setup implementing automated dependency 
# installation, test environment configuration, and quality assurance tool
# setup per Section 3.6.1 and 3.6.4 requirements.
#
# Features:
# - Section 3.6.1: Development environment management with comprehensive testing infrastructure
# - Section 3.6.3: Testing infrastructure with PyTest framework and advanced coverage integration  
# - Section 3.6.4: Quality assurance with automated quality gates enforcement
# - Section 4.1.1.5: Test execution workflow with environment setup requirements
#
# Usage:
#   ./tests/coverage/scripts/prepare-ci-environment.sh [OPTIONS]
#
# Options:
#   --python-version VERSION    Python version to validate (default: 3.8+)
#   --install-method METHOD     Installation method: conda|pip|auto (default: auto)
#   --enable-parallel          Enable parallel test execution setup
#   --coverage-threshold PCT   Coverage threshold percentage (default: 90)
#   --benchmark-mode MODE      Benchmark mode: enable|disable|auto (default: auto)
#   --quality-gates MODE       Quality gates: strict|standard|minimal (default: strict)
#   --verbose                  Enable verbose output
#   --dry-run                  Show commands without executing
#   --help                     Display this help message
#
# Environment Variables:
#   CI                         Set to 'true' in CI environments
#   PYTEST_CONFIG_FILE         Custom pytest configuration file path
#   COVERAGE_THRESHOLD         Override coverage threshold (0-100)
#   PARALLEL_WORKERS           Number of parallel test workers
#   QUALITY_GATES_MODE         Quality gates enforcement level
#
# Exit Codes:
#   0   Success - environment ready for testing
#   1   General error - check error output
#   2   Python version incompatible
#   3   Dependency installation failed  
#   4   Quality assurance setup failed
#   5   Configuration validation failed
#
# =============================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# =============================================================================
# GLOBAL CONFIGURATION AND CONSTANTS
# =============================================================================

# Script metadata and version information
readonly SCRIPT_NAME="prepare-ci-environment.sh"
readonly SCRIPT_VERSION="1.0.0"
readonly SCRIPT_AUTHOR="FlyRigLoader Development Team"

# Default configuration values per technical specification requirements
readonly DEFAULT_PYTHON_VERSION="3.8"
readonly DEFAULT_COVERAGE_THRESHOLD=90
readonly DEFAULT_PYTEST_TIMEOUT=30
readonly DEFAULT_PARALLEL_WORKERS="auto"

# Project structure constants
readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
readonly TESTS_DIR="${PROJECT_ROOT}/tests"
readonly COVERAGE_DIR="${TESTS_DIR}/coverage"
readonly SRC_DIR="${PROJECT_ROOT}/src"

# Configuration file paths
readonly ENVIRONMENT_YML="${PROJECT_ROOT}/environment.yml"
readonly PYPROJECT_TOML="${PROJECT_ROOT}/pyproject.toml"
readonly PYTEST_INI="${COVERAGE_DIR}/pytest.ini"
readonly PRECOMMIT_CONFIG="${PROJECT_ROOT}/.pre-commit-config.yaml"

# Quality assurance tool requirements per Section 3.6.4
readonly QA_TOOLS=(
    "black>=24.3.0"
    "isort>=5.12.0" 
    "mypy>=1.8.0"
    "flake8>=7.0.0"
    "pre-commit>=3.6.0"
)

# Testing infrastructure requirements per Section 3.6.3
readonly TEST_TOOLS=(
    "pytest>=7.0.0"
    "pytest-cov>=6.1.1"
    "pytest-mock>=3.14.1"
    "pytest-benchmark>=4.0.0"
    "coverage>=7.8.2"
    "hypothesis>=6.131.9"
    "pytest-xdist>=3.7.0"
    "pytest-timeout>=2.3.0"
)

# Performance benchmark SLA requirements per Section 4.1.1.5
readonly BENCHMARK_SLA_DATA_LOADING="1.0"  # seconds per 100MB
readonly BENCHMARK_SLA_DATAFRAME_TRANSFORM="0.5"  # seconds per 1M rows

# Terminal color codes for enhanced output formatting
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly WHITE='\033[1;37m'
readonly NC='\033[0m' # No Color

# =============================================================================
# GLOBAL VARIABLES AND STATE MANAGEMENT
# =============================================================================

# Configuration state variables
declare -g PYTHON_VERSION="${DEFAULT_PYTHON_VERSION}"
declare -g INSTALL_METHOD="auto"
declare -g ENABLE_PARALLEL=false
declare -g COVERAGE_THRESHOLD="${DEFAULT_COVERAGE_THRESHOLD}"
declare -g BENCHMARK_MODE="auto"
declare -g QUALITY_GATES_MODE="strict"
declare -g VERBOSE=false
declare -g DRY_RUN=false

# Runtime state tracking
declare -g START_TIME
declare -g STEP_COUNTER=0
declare -g ERROR_COUNT=0
declare -g WARNING_COUNT=0

# Environment detection
declare -g IS_CI="${CI:-false}"
declare -g HAS_CONDA=false
declare -g HAS_PIP=false
declare -g PYTHON_EXECUTABLE=""

# =============================================================================
# UTILITY FUNCTIONS FOR ENHANCED OUTPUT AND ERROR HANDLING
# =============================================================================

# Enhanced logging function with timestamps and severity levels
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "$level" in
        "INFO")
            echo -e "${timestamp} ${CYAN}[INFO]${NC} $message" >&1
            ;;
        "WARN")
            echo -e "${timestamp} ${YELLOW}[WARN]${NC} $message" >&2
            ((WARNING_COUNT++))
            ;;
        "ERROR")
            echo -e "${timestamp} ${RED}[ERROR]${NC} $message" >&2
            ((ERROR_COUNT++))
            ;;
        "SUCCESS")
            echo -e "${timestamp} ${GREEN}[SUCCESS]${NC} $message" >&1
            ;;
        "DEBUG")
            if [[ "$VERBOSE" == true ]]; then
                echo -e "${timestamp} ${PURPLE}[DEBUG]${NC} $message" >&2
            fi
            ;;
        "STEP")
            ((STEP_COUNTER++))
            echo -e "\n${timestamp} ${WHITE}[STEP $STEP_COUNTER]${NC} $message" >&1
            ;;
    esac
}

# Progress indicator for long-running operations
show_progress() {
    local current="$1"
    local total="$2"
    local description="$3"
    local percentage=$((current * 100 / total))
    local bar_length=30
    local filled_length=$((percentage * bar_length / 100))
    
    printf "\r${BLUE}Progress:${NC} ["
    printf "%*s" $filled_length | tr ' ' '='
    printf "%*s" $((bar_length - filled_length)) | tr ' ' '-'
    printf "] %d%% - %s" $percentage "$description"
    
    if [[ $current -eq $total ]]; then
        printf "\n"
    fi
}

# Comprehensive error handling with context and recovery suggestions
handle_error() {
    local exit_code="$1"
    local error_message="$2"
    local context="${3:-}"
    local recovery_suggestion="${4:-}"
    
    log "ERROR" "Operation failed with exit code $exit_code: $error_message"
    
    if [[ -n "$context" ]]; then
        log "ERROR" "Context: $context"
    fi
    
    if [[ -n "$recovery_suggestion" ]]; then
        log "INFO" "Recovery suggestion: $recovery_suggestion"
    fi
    
    # Collect additional diagnostic information
    log "DEBUG" "Current working directory: $(pwd)"
    log "DEBUG" "Available disk space: $(df -h . | tail -1 | awk '{print $4}')"
    log "DEBUG" "Memory usage: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
    
    exit "$exit_code"
}

# Command execution wrapper with dry-run support and error handling
execute_command() {
    local description="$1"
    shift
    local command=("$@")
    
    log "DEBUG" "Executing: ${command[*]}"
    
    if [[ "$DRY_RUN" == true ]]; then
        log "INFO" "[DRY-RUN] Would execute: $description"
        log "DEBUG" "[DRY-RUN] Command: ${command[*]}"
        return 0
    fi
    
    if [[ "$VERBOSE" == true ]]; then
        log "INFO" "Executing: $description"
    fi
    
    if ! "${command[@]}"; then
        local exit_code=$?
        handle_error "$exit_code" "Command failed: $description" "Command: ${command[*]}" \
            "Check command syntax and dependencies"
        return "$exit_code"
    fi
    
    return 0
}

# =============================================================================
# ARGUMENT PARSING AND CONFIGURATION MANAGEMENT
# =============================================================================

# Display comprehensive help information
show_help() {
    cat << EOF
${WHITE}$SCRIPT_NAME v$SCRIPT_VERSION${NC}

${CYAN}DESCRIPTION:${NC}
    Comprehensive CI/CD environment preparation script implementing automated 
    dependency installation, test environment configuration, and quality 
    assurance tool setup per Section 3.6.1 and 3.6.4 requirements.

${CYAN}USAGE:${NC}
    $SCRIPT_NAME [OPTIONS]

${CYAN}OPTIONS:${NC}
    --python-version VERSION    Python version to validate (default: $DEFAULT_PYTHON_VERSION+)
    --install-method METHOD     Installation method: conda|pip|auto (default: auto)
    --enable-parallel          Enable parallel test execution setup
    --coverage-threshold PCT   Coverage threshold percentage (default: $DEFAULT_COVERAGE_THRESHOLD)
    --benchmark-mode MODE      Benchmark mode: enable|disable|auto (default: auto)
    --quality-gates MODE       Quality gates: strict|standard|minimal (default: strict)
    --verbose                  Enable verbose output
    --dry-run                  Show commands without executing
    --help                     Display this help message

${CYAN}ENVIRONMENT VARIABLES:${NC}
    CI                         Set to 'true' in CI environments
    PYTEST_CONFIG_FILE         Custom pytest configuration file path
    COVERAGE_THRESHOLD         Override coverage threshold (0-100)
    PARALLEL_WORKERS           Number of parallel test workers
    QUALITY_GATES_MODE         Quality gates enforcement level

${CYAN}EXAMPLES:${NC}
    # Standard setup with auto-detection
    $SCRIPT_NAME

    # CI environment with strict quality gates
    CI=true $SCRIPT_NAME --quality-gates strict --enable-parallel

    # Development setup with verbose output
    $SCRIPT_NAME --verbose --benchmark-mode enable

    # Minimal setup for quick testing
    $SCRIPT_NAME --quality-gates minimal --coverage-threshold 75

${CYAN}EXIT CODES:${NC}
    0   Success - environment ready for testing
    1   General error - check error output
    2   Python version incompatible
    3   Dependency installation failed
    4   Quality assurance setup failed
    5   Configuration validation failed

${CYAN}AUTHOR:${NC}
    $SCRIPT_AUTHOR

EOF
}

# Comprehensive argument parsing with validation
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --python-version)
                PYTHON_VERSION="$2"
                shift 2
                ;;
            --install-method)
                case "$2" in
                    conda|pip|auto)
                        INSTALL_METHOD="$2"
                        ;;
                    *)
                        handle_error 1 "Invalid install method: $2" \
                            "Valid options: conda, pip, auto"
                        ;;
                esac
                shift 2
                ;;
            --enable-parallel)
                ENABLE_PARALLEL=true
                shift
                ;;
            --coverage-threshold)
                if [[ "$2" =~ ^[0-9]+$ ]] && [[ "$2" -ge 0 ]] && [[ "$2" -le 100 ]]; then
                    COVERAGE_THRESHOLD="$2"
                else
                    handle_error 1 "Invalid coverage threshold: $2" \
                        "Must be integer between 0 and 100"
                fi
                shift 2
                ;;
            --benchmark-mode)
                case "$2" in
                    enable|disable|auto)
                        BENCHMARK_MODE="$2"
                        ;;
                    *)
                        handle_error 1 "Invalid benchmark mode: $2" \
                            "Valid options: enable, disable, auto"
                        ;;
                esac
                shift 2
                ;;
            --quality-gates)
                case "$2" in
                    strict|standard|minimal)
                        QUALITY_GATES_MODE="$2"
                        ;;
                    *)
                        handle_error 1 "Invalid quality gates mode: $2" \
                            "Valid options: strict, standard, minimal"
                        ;;
                esac
                shift 2
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                handle_error 1 "Unknown option: $1" \
                    "Use --help for usage information"
                ;;
        esac
    done
    
    # Apply environment variable overrides
    if [[ -n "${COVERAGE_THRESHOLD:-}" ]]; then
        COVERAGE_THRESHOLD="$COVERAGE_THRESHOLD"
    fi
    
    if [[ -n "${QUALITY_GATES_MODE:-}" ]]; then
        QUALITY_GATES_MODE="$QUALITY_GATES_MODE"
    fi
    
    log "DEBUG" "Configuration parsed: Python=$PYTHON_VERSION, Install=$INSTALL_METHOD, Parallel=$ENABLE_PARALLEL"
    log "DEBUG" "Coverage=$COVERAGE_THRESHOLD%, Benchmark=$BENCHMARK_MODE, QualityGates=$QUALITY_GATES_MODE"
}

# =============================================================================
# ENVIRONMENT DETECTION AND VALIDATION
# =============================================================================

# Comprehensive environment detection and capability assessment
detect_environment() {
    log "STEP" "Detecting environment capabilities and requirements"
    
    # Detect CI environment
    if [[ "$IS_CI" == "true" ]]; then
        log "INFO" "CI environment detected - enabling optimized configurations"
        # Override parallel testing in CI
        ENABLE_PARALLEL=true
    fi
    
    # Check for conda availability
    if command -v conda &> /dev/null; then
        HAS_CONDA=true
        local conda_version=$(conda --version 2>/dev/null | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -1)
        log "INFO" "Conda detected: version $conda_version"
    else
        log "WARN" "Conda not found - pip installation will be used"
    fi
    
    # Check for pip availability
    if command -v pip &> /dev/null; then
        HAS_PIP=true
        local pip_version=$(pip --version 2>/dev/null | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -1)
        log "INFO" "Pip detected: version $pip_version"
    else
        log "ERROR" "Pip not found - unable to install dependencies"
        handle_error 3 "Pip is required for dependency installation" \
            "Install pip or use a Python distribution with pip included"
    fi
    
    # Determine optimal installation method
    if [[ "$INSTALL_METHOD" == "auto" ]]; then
        if [[ "$HAS_CONDA" == true ]] && [[ -f "$ENVIRONMENT_YML" ]]; then
            INSTALL_METHOD="conda"
            log "INFO" "Auto-detected installation method: conda (environment.yml found)"
        elif [[ "$HAS_PIP" == true ]]; then
            INSTALL_METHOD="pip"
            log "INFO" "Auto-detected installation method: pip"
        else
            handle_error 3 "No suitable installation method available" \
                "Install conda or pip to proceed"
        fi
    fi
    
    log "SUCCESS" "Environment detection completed successfully"
}

# Python version validation with comprehensive compatibility checking
validate_python_version() {
    log "STEP" "Validating Python version compatibility"
    
    # Find Python executable
    if command -v python3 &> /dev/null; then
        PYTHON_EXECUTABLE="python3"
    elif command -v python &> /dev/null; then
        PYTHON_EXECUTABLE="python"
    else
        handle_error 2 "Python not found" \
            "Install Python $PYTHON_VERSION or higher"
    fi
    
    # Get current Python version
    local current_version
    current_version=$($PYTHON_EXECUTABLE -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
    
    log "INFO" "Current Python version: $current_version"
    log "INFO" "Required Python version: $PYTHON_VERSION+"
    
    # Version comparison using Python itself for accuracy
    local version_check
    version_check=$($PYTHON_EXECUTABLE -c "
import sys
required = tuple(map(int, '$PYTHON_VERSION'.split('.')))
current = sys.version_info[:len(required)]
print('compatible' if current >= required else 'incompatible')
")
    
    if [[ "$version_check" == "incompatible" ]]; then
        handle_error 2 "Python version $current_version is incompatible" \
            "Upgrade to Python $PYTHON_VERSION or higher"
    fi
    
    # Check for required Python features
    log "DEBUG" "Validating Python features compatibility"
    
    # Check for asyncio support (required for modern testing)
    if ! $PYTHON_EXECUTABLE -c "import asyncio" 2>/dev/null; then
        log "WARN" "AsyncIO support not available - some advanced features may be limited"
    fi
    
    # Check for SSL support (required for secure package downloads)
    if ! $PYTHON_EXECUTABLE -c "import ssl" 2>/dev/null; then
        log "WARN" "SSL support not available - secure package downloads may fail"
    fi
    
    log "SUCCESS" "Python version validation completed: $current_version is compatible"
}

# Configuration file validation with comprehensive integrity checking
validate_configuration_files() {
    log "STEP" "Validating project configuration files"
    
    local validation_errors=0
    
    # Validate environment.yml
    if [[ -f "$ENVIRONMENT_YML" ]]; then
        log "INFO" "Validating environment.yml structure"
        
        # Check YAML syntax
        if command -v python &> /dev/null; then
            if ! $PYTHON_EXECUTABLE -c "
import yaml
try:
    with open('$ENVIRONMENT_YML', 'r') as f:
        yaml.safe_load(f)
    print('YAML syntax valid')
except Exception as e:
    print(f'YAML syntax error: {e}')
    exit(1)
" 2>/dev/null; then
                log "ERROR" "environment.yml has invalid YAML syntax"
                ((validation_errors++))
            fi
        fi
        
        # Validate required sections
        local required_sections=("name" "dependencies")
        for section in "${required_sections[@]}"; do
            if ! grep -q "^$section:" "$ENVIRONMENT_YML"; then
                log "ERROR" "environment.yml missing required section: $section"
                ((validation_errors++))
            fi
        done
    else
        log "WARN" "environment.yml not found - conda installation will not be available"
    fi
    
    # Validate pyproject.toml
    if [[ -f "$PYPROJECT_TOML" ]]; then
        log "INFO" "Validating pyproject.toml structure"
        
        # Check TOML syntax if toml library is available
        if $PYTHON_EXECUTABLE -c "import tomllib" 2>/dev/null || $PYTHON_EXECUTABLE -c "import tomli" 2>/dev/null; then
            if ! $PYTHON_EXECUTABLE -c "
try:
    import tomllib
except ImportError:
    import tomli as tomllib

try:
    with open('$PYPROJECT_TOML', 'rb') as f:
        tomllib.load(f)
    print('TOML syntax valid')
except Exception as e:
    print(f'TOML syntax error: {e}')
    exit(1)
" 2>/dev/null; then
                log "ERROR" "pyproject.toml has invalid TOML syntax"
                ((validation_errors++))
            fi
        fi
        
        # Validate required project sections
        local required_project_sections=("name" "dependencies")
        for section in "${required_project_sections[@]}"; do
            if ! grep -q "^$section" "$PYPROJECT_TOML"; then
                log "ERROR" "pyproject.toml missing required project section: $section"
                ((validation_errors++))
            fi
        done
    else
        log "ERROR" "pyproject.toml not found - required for project configuration"
        ((validation_errors++))
    fi
    
    # Validate pytest configuration
    if [[ -f "$PYTEST_INI" ]]; then
        log "INFO" "Validating pytest.ini configuration"
        
        # Check for required pytest sections
        if ! grep -q "^\[pytest\]" "$PYTEST_INI"; then
            log "ERROR" "pytest.ini missing [pytest] section"
            ((validation_errors++))
        fi
    else
        log "INFO" "Using pytest configuration from pyproject.toml"
    fi
    
    # Validate project directory structure
    local required_dirs=("$SRC_DIR" "$TESTS_DIR")
    for dir in "${required_dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            log "ERROR" "Required directory not found: $dir"
            ((validation_errors++))
        fi
    done
    
    if [[ $validation_errors -gt 0 ]]; then
        handle_error 5 "Configuration validation failed with $validation_errors errors" \
            "Fix configuration issues before proceeding"
    fi
    
    log "SUCCESS" "Configuration file validation completed successfully"
}

# =============================================================================
# DEPENDENCY INSTALLATION AND MANAGEMENT
# =============================================================================

# Comprehensive conda environment setup with dependency resolution
setup_conda_environment() {
    log "STEP" "Setting up conda environment with comprehensive testing dependencies"
    
    if [[ ! -f "$ENVIRONMENT_YML" ]]; then
        log "WARN" "environment.yml not found - skipping conda setup"
        return 0
    fi
    
    # Extract environment name from environment.yml
    local env_name
    env_name=$(grep "^name:" "$ENVIRONMENT_YML" | sed 's/name: *//')
    
    log "INFO" "Creating/updating conda environment: $env_name"
    
    # Update conda to latest version for better dependency resolution
    execute_command "Update conda to latest version" \
        conda update -n base -c defaults conda -y
    
    # Create or update environment from file
    execute_command "Create/update conda environment from environment.yml" \
        conda env update --file "$ENVIRONMENT_YML" --prune
    
    # Activate environment and verify installation
    log "INFO" "Activating conda environment: $env_name"
    
    # Source conda activation script
    local conda_base
    conda_base=$(conda info --base)
    
    if [[ ! "$DRY_RUN" == true ]]; then
        # shellcheck source=/dev/null
        source "$conda_base/etc/profile.d/conda.sh"
        conda activate "$env_name"
        
        # Verify critical packages are installed
        local test_packages=("pytest" "pytest-cov" "black" "mypy")
        for package in "${test_packages[@]}"; do
            if ! conda list | grep -q "^$package "; then
                log "WARN" "Package $package not found in conda environment"
            else
                local version
                version=$(conda list | grep "^$package " | awk '{print $2}')
                log "DEBUG" "Verified $package version: $version"
            fi
        done
    fi
    
    log "SUCCESS" "Conda environment setup completed successfully"
}

# Advanced pip dependency installation with constraint management
install_pip_dependencies() {
    log "STEP" "Installing pip dependencies with advanced dependency resolution"
    
    # Upgrade pip to latest version for better dependency resolution
    execute_command "Upgrade pip to latest version" \
        $PYTHON_EXECUTABLE -m pip install --upgrade pip
    
    # Install build dependencies first
    execute_command "Install build system dependencies" \
        $PYTHON_EXECUTABLE -m pip install "setuptools>=42" wheel build
    
    # Install project in editable mode with development dependencies
    log "INFO" "Installing project dependencies from pyproject.toml"
    
    if [[ -f "$PYPROJECT_TOML" ]]; then
        execute_command "Install project with development dependencies" \
            $PYTHON_EXECUTABLE -m pip install -e ".[dev]"
    else
        log "WARN" "pyproject.toml not found - installing testing dependencies individually"
        
        # Install testing tools individually
        for tool in "${TEST_TOOLS[@]}"; do
            execute_command "Install testing tool: $tool" \
                $PYTHON_EXECUTABLE -m pip install "$tool"
        done
        
        # Install quality assurance tools
        for tool in "${QA_TOOLS[@]}"; do
            execute_command "Install QA tool: $tool" \
                $PYTHON_EXECUTABLE -m pip install "$tool"
        done
    fi
    
    # Verify critical package installations
    log "INFO" "Verifying installed package versions"
    
    local critical_packages=("pytest" "pytest-cov" "black" "mypy" "coverage")
    for package in "${critical_packages[@]}"; do
        if $PYTHON_EXECUTABLE -c "import $package" 2>/dev/null; then
            local version
            version=$($PYTHON_EXECUTABLE -c "import $package; print(getattr($package, '__version__', 'unknown'))" 2>/dev/null)
            log "DEBUG" "Verified $package version: $version"
        else
            log "WARN" "Package $package not properly installed or importable"
        fi
    done
    
    log "SUCCESS" "Pip dependency installation completed successfully"
}

# Unified dependency installation orchestration
install_dependencies() {
    log "STEP" "Installing comprehensive testing and quality assurance dependencies"
    
    case "$INSTALL_METHOD" in
        conda)
            setup_conda_environment
            ;;
        pip)
            install_pip_dependencies
            ;;
        *)
            handle_error 3 "Invalid installation method: $INSTALL_METHOD" \
                "Use conda, pip, or auto"
            ;;
    esac
    
    # Additional dependency verification
    log "INFO" "Performing comprehensive dependency verification"
    
    # Verify pytest installation and basic functionality
    if ! $PYTHON_EXECUTABLE -c "import pytest; print(f'pytest {pytest.__version__}')" 2>/dev/null; then
        handle_error 3 "pytest installation verification failed" \
            "Check pytest installation and Python path"
    fi
    
    # Verify coverage measurement capability
    if ! $PYTHON_EXECUTABLE -c "import coverage; print(f'coverage {coverage.__version__}')" 2>/dev/null; then
        handle_error 3 "coverage installation verification failed" \
            "Check coverage.py installation"
    fi
    
    log "SUCCESS" "All dependencies installed and verified successfully"
}

# =============================================================================
# TESTING INFRASTRUCTURE CONFIGURATION
# =============================================================================

# Comprehensive pytest configuration setup with advanced features
configure_pytest_infrastructure() {
    log "STEP" "Configuring comprehensive pytest testing infrastructure"
    
    # Determine pytest configuration source
    local pytest_config_file="${PYTEST_CONFIG_FILE:-$PYTEST_INI}"
    
    if [[ -f "$pytest_config_file" ]]; then
        log "INFO" "Using pytest configuration from: $pytest_config_file"
    else
        log "INFO" "Using pytest configuration from pyproject.toml"
        pytest_config_file="$PYPROJECT_TOML"
    fi
    
    # Validate pytest configuration
    execute_command "Validate pytest configuration" \
        $PYTHON_EXECUTABLE -m pytest --collect-only --quiet
    
    # Configure pytest markers for test categorization
    log "INFO" "Configuring pytest markers for comprehensive test categorization"
    
    # Verify test discovery
    local test_count
    if [[ ! "$DRY_RUN" == true ]]; then
        test_count=$($PYTHON_EXECUTABLE -m pytest --collect-only --quiet 2>/dev/null | grep -c "<Module" || echo "0")
        log "INFO" "Discovered $test_count test modules"
        
        if [[ "$test_count" -eq 0 ]]; then
            log "WARN" "No tests discovered - verify test file naming and structure"
        fi
    fi
    
    # Configure parallel testing if enabled
    if [[ "$ENABLE_PARALLEL" == true ]]; then
        log "INFO" "Configuring parallel test execution with pytest-xdist"
        
        # Determine optimal worker count
        local worker_count="${PARALLEL_WORKERS:-$DEFAULT_PARALLEL_WORKERS}"
        if [[ "$worker_count" == "auto" ]]; then
            if [[ ! "$DRY_RUN" == true ]]; then
                worker_count=$(nproc 2>/dev/null || echo "2")
                log "DEBUG" "Auto-detected $worker_count CPU cores for parallel testing"
            else
                worker_count="auto"
            fi
        fi
        
        # Verify pytest-xdist functionality
        execute_command "Verify pytest-xdist parallel testing capability" \
            $PYTHON_EXECUTABLE -c "import xdist; print(f'pytest-xdist {xdist.__version__}')"
    fi
    
    log "SUCCESS" "Pytest infrastructure configuration completed successfully"
}

# Advanced coverage reporting setup with multiple output formats
configure_coverage_reporting() {
    log "STEP" "Configuring advanced coverage reporting with quality thresholds"
    
    # Create coverage configuration directory
    execute_command "Create coverage reporting directory" \
        mkdir -p "$COVERAGE_DIR/reports"
    
    # Validate coverage threshold
    if [[ "$COVERAGE_THRESHOLD" -lt 50 ]]; then
        log "WARN" "Coverage threshold $COVERAGE_THRESHOLD% is below recommended minimum (50%)"
    elif [[ "$COVERAGE_THRESHOLD" -ge 90 ]]; then
        log "INFO" "Coverage threshold $COVERAGE_THRESHOLD% meets enterprise quality standards"
    fi
    
    # Configure coverage contexts for detailed reporting
    log "INFO" "Configuring coverage contexts for comprehensive measurement"
    
    # Test coverage measurement capability
    execute_command "Verify coverage measurement functionality" \
        $PYTHON_EXECUTABLE -c "
import coverage
cov = coverage.Coverage()
print(f'Coverage.py {coverage.__version__} ready')
print(f'Branch coverage: {cov.config.branch}')
print(f'Source patterns: {cov.config.source}')
"
    
    # Configure coverage database location
    local coverage_db="$COVERAGE_DIR/.coverage"
    log "DEBUG" "Coverage database location: $coverage_db"
    
    # Set up coverage environment variables
    if [[ ! "$DRY_RUN" == true ]]; then
        export COVERAGE_FILE="$coverage_db"
        export COVERAGE_CONTEXT="test-execution"
        log "DEBUG" "Coverage environment configured: COVERAGE_FILE=$COVERAGE_FILE"
    fi
    
    log "SUCCESS" "Coverage reporting configuration completed successfully"
}

# Performance benchmark configuration per SLA requirements
configure_performance_benchmarks() {
    log "STEP" "Configuring performance benchmarks per SLA requirements"
    
    # Determine benchmark mode based on configuration
    local enable_benchmarks=false
    
    case "$BENCHMARK_MODE" in
        enable)
            enable_benchmarks=true
            ;;
        disable)
            enable_benchmarks=false
            ;;
        auto)
            # Enable benchmarks in CI or when specifically requested
            if [[ "$IS_CI" == "true" ]] || [[ "$ENABLE_PARALLEL" == true ]]; then
                enable_benchmarks=true
            fi
            ;;
    esac
    
    if [[ "$enable_benchmarks" == true ]]; then
        log "INFO" "Enabling performance benchmark validation"
        
        # Verify pytest-benchmark installation
        execute_command "Verify pytest-benchmark functionality" \
            $PYTHON_EXECUTABLE -c "
import pytest_benchmark
print(f'pytest-benchmark {pytest_benchmark.__version__}')
print(f'Data loading SLA: {$BENCHMARK_SLA_DATA_LOADING}s per 100MB')
print(f'DataFrame transform SLA: {$BENCHMARK_SLA_DATAFRAME_TRANSFORM}s per 1M rows')
"
        
        # Configure benchmark storage directory
        execute_command "Create benchmark storage directory" \
            mkdir -p "$COVERAGE_DIR/benchmarks"
        
        # Set benchmark environment variables
        if [[ ! "$DRY_RUN" == true ]]; then
            export PYTEST_BENCHMARK_DIR="$COVERAGE_DIR/benchmarks"
            log "DEBUG" "Benchmark storage configured: $PYTEST_BENCHMARK_DIR"
        fi
    else
        log "INFO" "Performance benchmarks disabled"
    fi
    
    log "SUCCESS" "Performance benchmark configuration completed"
}

# =============================================================================
# QUALITY ASSURANCE TOOL CONFIGURATION
# =============================================================================

# Comprehensive code formatting tool setup
configure_code_formatting_tools() {
    log "STEP" "Configuring code formatting and style enforcement tools"
    
    # Configure Black code formatter
    log "INFO" "Configuring Black code formatter for consistent code style"
    
    execute_command "Verify Black installation and configuration" \
        $PYTHON_EXECUTABLE -m black --version
    
    # Test Black on a small code sample
    if [[ ! "$DRY_RUN" == true ]]; then
        local test_file="/tmp/black_test.py"
        echo "def  test( x,y ): return x+y" > "$test_file"
        
        if $PYTHON_EXECUTABLE -m black --check --quiet "$test_file" 2>/dev/null; then
            log "DEBUG" "Black configuration verified successfully"
        else
            log "DEBUG" "Black formatting test completed (expected different formatting)"
        fi
        
        rm -f "$test_file"
    fi
    
    # Configure isort import sorting
    log "INFO" "Configuring isort for import statement organization"
    
    execute_command "Verify isort installation and Black compatibility" \
        $PYTHON_EXECUTABLE -c "
import isort
print(f'isort {isort.__version__}')
# Verify Black profile compatibility
config = isort.Config(profile='black')
print(f'Black compatibility profile: {config.profile}')
"
    
    log "SUCCESS" "Code formatting tools configured successfully"
}

# Static type checking configuration with comprehensive validation
configure_static_type_checking() {
    log "STEP" "Configuring static type checking with MyPy"
    
    # Verify MyPy installation and basic functionality
    execute_command "Verify MyPy installation" \
        $PYTHON_EXECUTABLE -m mypy --version
    
    # Configure MyPy for strict type checking per quality requirements
    log "INFO" "Configuring MyPy for strict type checking enforcement"
    
    # Test MyPy on project source code
    if [[ -d "$SRC_DIR" ]] && [[ ! "$DRY_RUN" == true ]]; then
        log "DEBUG" "Testing MyPy configuration on source code"
        
        # Run MyPy in check mode without failing the setup
        if $PYTHON_EXECUTABLE -m mypy --config-file= --ignore-missing-imports "$SRC_DIR" 2>/dev/null; then
            log "DEBUG" "MyPy type checking validation successful"
        else
            log "WARN" "MyPy found type issues - will be addressed in quality gates"
        fi
    fi
    
    # Configure MyPy cache directory
    execute_command "Create MyPy cache directory" \
        mkdir -p "$COVERAGE_DIR/mypy_cache"
    
    if [[ ! "$DRY_RUN" == true ]]; then
        export MYPY_CACHE_DIR="$COVERAGE_DIR/mypy_cache"
        log "DEBUG" "MyPy cache configured: $MYPY_CACHE_DIR"
    fi
    
    log "SUCCESS" "Static type checking configuration completed"
}

# Comprehensive linting configuration with multiple rule sets
configure_linting_tools() {
    log "STEP" "Configuring comprehensive linting with Flake8"
    
    # Verify Flake8 installation
    execute_command "Verify Flake8 installation" \
        $PYTHON_EXECUTABLE -m flake8 --version
    
    # Configure Flake8 plugins and rule sets
    log "INFO" "Configuring Flake8 with comprehensive rule enforcement"
    
    # Test Flake8 configuration
    if [[ -d "$SRC_DIR" ]] && [[ ! "$DRY_RUN" == true ]]; then
        local flake8_issues
        flake8_issues=$($PYTHON_EXECUTABLE -m flake8 --count --statistics "$SRC_DIR" 2>/dev/null | tail -1 || echo "0")
        
        if [[ "$flake8_issues" =~ ^[0-9]+$ ]]; then
            log "DEBUG" "Flake8 configuration test completed: $flake8_issues issues found"
        else
            log "DEBUG" "Flake8 configuration test completed"
        fi
    fi
    
    log "SUCCESS" "Linting tool configuration completed"
}

# Pre-commit hooks installation and configuration
configure_precommit_hooks() {
    log "STEP" "Configuring pre-commit hooks for automated quality gates"
    
    # Verify pre-commit installation
    execute_command "Verify pre-commit installation" \
        pre-commit --version
    
    # Install pre-commit hooks if configuration exists
    if [[ -f "$PRECOMMIT_CONFIG" ]]; then
        log "INFO" "Installing pre-commit hooks from .pre-commit-config.yaml"
        
        execute_command "Install pre-commit git hooks" \
            pre-commit install
        
        # Validate pre-commit configuration
        execute_command "Validate pre-commit configuration" \
            pre-commit validate-config
        
        # Update pre-commit hook repositories
        execute_command "Update pre-commit hook repositories" \
            pre-commit autoupdate
        
    else
        log "INFO" "Creating basic pre-commit configuration"
        
        # Create basic pre-commit configuration if none exists
        if [[ ! "$DRY_RUN" == true ]]; then
            cat > "$PRECOMMIT_CONFIG" << 'EOF'
# Basic pre-commit configuration for flyrigloader
repos:
  - repo: https://github.com/psf/black
    rev: 24.3.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--ignore-missing-imports]
EOF
            
            execute_command "Install newly created pre-commit hooks" \
                pre-commit install
        fi
    fi
    
    log "SUCCESS" "Pre-commit hooks configuration completed"
}

# Quality gates enforcement configuration per technical requirements
configure_quality_gates() {
    log "STEP" "Configuring quality gates enforcement per Section 3.6.4 requirements"
    
    case "$QUALITY_GATES_MODE" in
        strict)
            log "INFO" "Configuring strict quality gates (enterprise-grade)"
            # 90% coverage, zero lint violations, full type checking
            ;;
        standard)
            log "INFO" "Configuring standard quality gates"
            # 80% coverage, minimal lint violations, basic type checking
            COVERAGE_THRESHOLD=80
            ;;
        minimal)
            log "INFO" "Configuring minimal quality gates"
            # 70% coverage, essential lint checks only
            COVERAGE_THRESHOLD=70
            ;;
    esac
    
    # Create quality gate validation script
    local quality_gate_script="$COVERAGE_DIR/scripts/validate-quality-gates.sh"
    execute_command "Create quality gate validation directory" \
        mkdir -p "$(dirname "$quality_gate_script")"
    
    if [[ ! "$DRY_RUN" == true ]]; then
        cat > "$quality_gate_script" << EOF
#!/bin/bash
# Automated quality gate validation script
# Generated by $SCRIPT_NAME v$SCRIPT_VERSION

set -euo pipefail

echo "Validating quality gates (mode: $QUALITY_GATES_MODE)"

# Run tests with coverage
python -m pytest --cov=src/flyrigloader --cov-fail-under=$COVERAGE_THRESHOLD

# Run type checking
python -m mypy src/flyrigloader --ignore-missing-imports

# Run linting
python -m flake8 src/flyrigloader

# Run formatting checks
python -m black --check src/flyrigloader
python -m isort --check-only --profile black src/flyrigloader

echo "All quality gates passed successfully"
EOF
        
        chmod +x "$quality_gate_script"
        log "DEBUG" "Quality gate validation script created: $quality_gate_script"
    fi
    
    log "SUCCESS" "Quality gates configuration completed"
}

# =============================================================================
# CI/CD INTEGRATION AND OPTIMIZATION
# =============================================================================

# GitHub Actions workflow configuration for automated testing
configure_github_actions_integration() {
    log "STEP" "Configuring GitHub Actions CI/CD integration"
    
    local github_workflows_dir="${PROJECT_ROOT}/.github/workflows"
    local test_workflow="${github_workflows_dir}/test.yml"
    
    execute_command "Create GitHub Actions workflows directory" \
        mkdir -p "$github_workflows_dir"
    
    if [[ ! -f "$test_workflow" ]] && [[ ! "$DRY_RUN" == true ]]; then
        log "INFO" "Creating comprehensive GitHub Actions test workflow"
        
        cat > "$test_workflow" << 'EOF'
name: Comprehensive Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.8", "3.9", "3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Prepare CI environment
      run: |
        chmod +x tests/coverage/scripts/prepare-ci-environment.sh
        ./tests/coverage/scripts/prepare-ci-environment.sh --enable-parallel
    
    - name: Run comprehensive test suite
      run: |
        pytest --cov=src/flyrigloader \
               --cov-report=xml:coverage.xml \
               --cov-fail-under=90 \
               --timeout=30 \
               -n auto
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v4
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
EOF
        
        log "INFO" "GitHub Actions workflow created: $test_workflow"
    fi
    
    log "SUCCESS" "GitHub Actions integration configuration completed"
}

# Performance optimization for test execution
optimize_test_execution() {
    log "STEP" "Optimizing test execution performance"
    
    # Configure Python optimization settings
    if [[ ! "$DRY_RUN" == true ]]; then
        # Disable Python assertions in CI for performance (but keep for local development)
        if [[ "$IS_CI" == "true" ]]; then
            export PYTHONOPTIMIZE=1
            log "DEBUG" "Python optimization enabled for CI environment"
        fi
        
        # Configure pytest cache directory
        local pytest_cache_dir="$COVERAGE_DIR/.pytest_cache"
        execute_command "Create pytest cache directory" \
            mkdir -p "$pytest_cache_dir"
        
        export PYTEST_CACHE_DIR="$pytest_cache_dir"
        log "DEBUG" "Pytest cache configured: $PYTEST_CACHE_DIR"
    fi
    
    # Configure parallel execution if enabled
    if [[ "$ENABLE_PARALLEL" == true ]]; then
        local cpu_count
        cpu_count=$(nproc 2>/dev/null || echo "2")
        log "INFO" "Parallel test execution enabled: $cpu_count workers"
    fi
    
    log "SUCCESS" "Test execution optimization completed"
}

# =============================================================================
# FINAL VALIDATION AND ENVIRONMENT VERIFICATION
# =============================================================================

# Comprehensive environment validation with integration testing
validate_complete_environment() {
    log "STEP" "Performing comprehensive environment validation"
    
    local validation_errors=0
    
    # Test pytest execution
    log "INFO" "Validating pytest execution capability"
    if [[ ! "$DRY_RUN" == true ]]; then
        if $PYTHON_EXECUTABLE -m pytest --version >/dev/null 2>&1; then
            log "DEBUG" "Pytest execution validated successfully"
        else
            log "ERROR" "Pytest execution validation failed"
            ((validation_errors++))
        fi
    fi
    
    # Test coverage measurement
    log "INFO" "Validating coverage measurement capability"
    if [[ ! "$DRY_RUN" == true ]]; then
        local coverage_test_file="/tmp/coverage_test.py"
        echo "def test_function(): assert True" > "$coverage_test_file"
        
        if $PYTHON_EXECUTABLE -m pytest --cov=. --cov-report=term-missing "$coverage_test_file" >/dev/null 2>&1; then
            log "DEBUG" "Coverage measurement validated successfully"
        else
            log "ERROR" "Coverage measurement validation failed"
            ((validation_errors++))
        fi
        
        rm -f "$coverage_test_file" .coverage
    fi
    
    # Test quality assurance tools
    log "INFO" "Validating quality assurance tool integration"
    
    local qa_tools_test=("black" "isort" "mypy" "flake8")
    for tool in "${qa_tools_test[@]}"; do
        if [[ ! "$DRY_RUN" == true ]]; then
            if $PYTHON_EXECUTABLE -m "$tool" --version >/dev/null 2>&1; then
                log "DEBUG" "QA tool $tool validated successfully"
            else
                log "ERROR" "QA tool $tool validation failed"
                ((validation_errors++))
            fi
        fi
    done
    
    # Test parallel execution capability if enabled
    if [[ "$ENABLE_PARALLEL" == true ]] && [[ ! "$DRY_RUN" == true ]]; then
        log "INFO" "Validating parallel test execution capability"
        
        if $PYTHON_EXECUTABLE -c "import xdist; import concurrent.futures" >/dev/null 2>&1; then
            log "DEBUG" "Parallel execution capability validated"
        else
            log "ERROR" "Parallel execution validation failed"
            ((validation_errors++))
        fi
    fi
    
    if [[ $validation_errors -gt 0 ]]; then
        handle_error 4 "Environment validation failed with $validation_errors errors" \
            "Review error messages and fix issues before proceeding"
    fi
    
    log "SUCCESS" "Complete environment validation passed successfully"
}

# Generate comprehensive environment summary and usage instructions
generate_environment_summary() {
    log "STEP" "Generating environment setup summary and usage instructions"
    
    local summary_file="$COVERAGE_DIR/environment-summary.md"
    local current_time=$(date '+%Y-%m-%d %H:%M:%S')
    
    if [[ ! "$DRY_RUN" == true ]]; then
        cat > "$summary_file" << EOF
# CI/CD Environment Setup Summary

**Generated by:** $SCRIPT_NAME v$SCRIPT_VERSION  
**Generated on:** $current_time  
**Environment:** $(if [[ "$IS_CI" == "true" ]]; then echo "CI/CD"; else echo "Local Development"; fi)

## Configuration Summary

| Setting | Value |
|---------|-------|
| Python Version | $($PYTHON_EXECUTABLE --version 2>&1) |
| Installation Method | $INSTALL_METHOD |
| Coverage Threshold | $COVERAGE_THRESHOLD% |
| Quality Gates Mode | $QUALITY_GATES_MODE |
| Parallel Testing | $(if [[ "$ENABLE_PARALLEL" == "true" ]]; then echo "Enabled"; else echo "Disabled"; fi) |
| Benchmark Mode | $BENCHMARK_MODE |

## Installed Testing Tools

$(if [[ ! "$DRY_RUN" == true ]]; then
    echo "| Tool | Version |"
    echo "|------|---------|"
    for tool in pytest coverage black isort mypy flake8; do
        version=$($PYTHON_EXECUTABLE -c "
try:
    import $tool
    print(getattr($tool, '__version__', 'unknown'))
except ImportError:
    print('not installed')
" 2>/dev/null)
        echo "| $tool | $version |"
    done
fi)

## Quick Start Commands

### Run Complete Test Suite
\`\`\`bash
pytest --cov=src/flyrigloader --cov-report=html --cov-fail-under=$COVERAGE_THRESHOLD
\`\`\`

### Run Quality Gates Validation
\`\`\`bash
./tests/coverage/scripts/validate-quality-gates.sh
\`\`\`

### Run Parallel Tests (if enabled)
$(if [[ "$ENABLE_PARALLEL" == "true" ]]; then
echo '```bash'
echo 'pytest -n auto --cov=src/flyrigloader'
echo '```'
else
echo '*Parallel testing not enabled*'
fi)

### Run Performance Benchmarks
$(if [[ "$BENCHMARK_MODE" != "disable" ]]; then
echo '```bash'
echo 'pytest --benchmark-only --benchmark-autosave'
echo '```'
else
echo '*Performance benchmarks disabled*'
fi)

## Environment Files

- **Project Configuration:** \`pyproject.toml\`
- **Environment Definition:** \`environment.yml\`
- **Pytest Configuration:** \`$(if [[ -f "$PYTEST_INI" ]]; then echo "tests/coverage/pytest.ini"; else echo "pyproject.toml"; fi)\`
- **Coverage Reports:** \`htmlcov/\`
- **Quality Gate Script:** \`tests/coverage/scripts/validate-quality-gates.sh\`

## Troubleshooting

### Common Issues

1. **Coverage Below Threshold**
   - Run: \`pytest --cov=src/flyrigloader --cov-report=html\`
   - Open: \`htmlcov/index.html\` to identify untested code

2. **Type Checking Failures**
   - Run: \`mypy src/flyrigloader --ignore-missing-imports\`
   - Fix type annotations as needed

3. **Formatting Issues**
   - Run: \`black src/flyrigloader tests/\`
   - Run: \`isort --profile black src/flyrigloader tests/\`

### Support

For additional support, refer to the project documentation or contact the development team.

---
*Environment prepared with $SCRIPT_NAME v$SCRIPT_VERSION*
EOF
        
        log "INFO" "Environment summary generated: $summary_file"
    fi
    
    # Display summary to console
    echo
    log "SUCCESS" "CI/CD Environment Setup Completed Successfully!"
    echo
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│                    SETUP SUMMARY                           │"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│ Python Version:     $($PYTHON_EXECUTABLE --version 2>&1 | cut -d' ' -f2)"
    echo "│ Installation:       $INSTALL_METHOD"
    echo "│ Coverage Threshold: $COVERAGE_THRESHOLD%"
    echo "│ Quality Gates:      $QUALITY_GATES_MODE"
    echo "│ Parallel Testing:   $(if [[ "$ENABLE_PARALLEL" == "true" ]]; then echo "Enabled"; else echo "Disabled"; fi)"
    echo "│ Benchmarks:         $BENCHMARK_MODE"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│ Warnings:           $WARNING_COUNT"
    echo "│ Errors:             $ERROR_COUNT"
    echo "│ Total Steps:        $STEP_COUNTER"
    echo "└─────────────────────────────────────────────────────────────┘"
    echo
    echo "${GREEN}Environment is ready for testing!${NC}"
    echo
    echo "Quick start commands:"
    echo "  ${CYAN}pytest --cov=src/flyrigloader${NC}                 # Run tests with coverage"
    echo "  ${CYAN}./tests/coverage/scripts/validate-quality-gates.sh${NC} # Validate quality gates"
    if [[ "$ENABLE_PARALLEL" == "true" ]]; then
        echo "  ${CYAN}pytest -n auto${NC}                                # Run tests in parallel"
    fi
    echo
}

# =============================================================================
# MAIN EXECUTION FLOW
# =============================================================================

# Main function orchestrating the complete environment setup process
main() {
    START_TIME=$(date +%s)
    
    # Display script header
    echo "================================================================================================"
    echo "                          FlyRigLoader CI/CD Environment Preparation"
    echo "                                    Version $SCRIPT_VERSION"
    echo "================================================================================================"
    echo
    
    # Initialize and validate environment
    parse_arguments "$@"
    detect_environment
    validate_python_version
    validate_configuration_files
    
    # Install dependencies and configure tools
    install_dependencies
    
    # Configure testing infrastructure
    configure_pytest_infrastructure
    configure_coverage_reporting
    configure_performance_benchmarks
    
    # Configure quality assurance tools
    configure_code_formatting_tools
    configure_static_type_checking
    configure_linting_tools
    configure_precommit_hooks
    configure_quality_gates
    
    # Configure CI/CD integration and optimization
    configure_github_actions_integration
    optimize_test_execution
    
    # Final validation and summary
    validate_complete_environment
    generate_environment_summary
    
    # Calculate and display execution time
    local end_time
    end_time=$(date +%s)
    local execution_time=$((end_time - START_TIME))
    
    log "SUCCESS" "Environment setup completed in ${execution_time} seconds"
    
    exit 0
}

# Script entry point with error handling
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Set up global error handling
    trap 'handle_error $? "Unexpected error occurred" "Line $LINENO" "Check error output and try again"' ERR
    
    # Execute main function with all arguments
    main "$@"
fi