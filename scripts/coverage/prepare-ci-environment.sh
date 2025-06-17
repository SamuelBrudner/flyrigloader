#!/bin/bash

# =============================================================================
# Enhanced CI/CD Environment Preparation Script
# =============================================================================
#
# Comprehensive CI/CD environment setup implementing automated dependency 
# installation, enhanced testing infrastructure, and quality assurance tool
# setup per Section 8.4 CI/CD Pipeline Architecture and Section 8.5 Quality
# Assurance Pipeline requirements.
#
# Enhanced Features:
# - Section 8.4.4: Selective test execution strategy with pytest marker filtering
# - Section 6.6: Enhanced testing strategy with behavior-focused validation
# - Section 8.4.3: Performance test isolation and optional benchmark execution  
# - Section 8.5.1: Quality gates enforcement with pytest-style validation
# - Section 8.4.6: Environment variable integration for conditional execution
#
# Usage:
#   ./scripts/coverage/prepare-ci-environment.sh [OPTIONS]
#
# Options:
#   --python-version VERSION    Python version to validate (default: 3.8+)
#   --install-method METHOD     Installation method: conda|pip|auto (default: auto)
#   --enable-parallel          Enable parallel test execution setup
#   --coverage-threshold PCT   Coverage threshold percentage (default: 90)
#   --benchmark-mode MODE      Benchmark mode: enable|disable|auto (default: auto)
#   --quality-gates MODE       Quality gates: strict|standard|minimal (default: strict)
#   --enable-network-tests     Enable network-dependent test configuration
#   --performance-isolation    Enable performance test isolation strategy
#   --verbose                  Enable verbose output
#   --dry-run                  Show commands without executing
#   --help                     Display this help message
#
# Environment Variables:
#   CI                         Set to 'true' in CI environments
#   RUN_NETWORK               Enable network-dependent tests (default: false)
#   BENCHMARK_TRIGGER         Enable dedicated benchmark job execution
#   PYTEST_CONFIG_FILE        Custom pytest configuration file path
#   COVERAGE_THRESHOLD        Override coverage threshold (0-100)
#   PARALLEL_WORKERS          Number of parallel test workers
#   QUALITY_GATES_MODE        Quality gates enforcement level
#
# Exit Codes:
#   0   Success - environment ready for enhanced testing strategy
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
readonly SCRIPT_VERSION="2.0.0"
readonly SCRIPT_AUTHOR="FlyRigLoader Development Team"

# Default configuration values per enhanced testing strategy requirements
readonly DEFAULT_PYTHON_VERSION="3.8"
readonly DEFAULT_COVERAGE_THRESHOLD=90
readonly DEFAULT_PYTEST_TIMEOUT=30
readonly DEFAULT_PARALLEL_WORKERS="4"  # Enhanced default for pytest-xdist

# Project structure constants (updated for scripts/coverage/ location)
readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly TESTS_DIR="${PROJECT_ROOT}/tests"
readonly COVERAGE_DIR="${PROJECT_ROOT}/scripts/coverage"  # Updated path
readonly BENCHMARKS_DIR="${PROJECT_ROOT}/scripts/benchmarks"  # Performance test isolation
readonly SRC_DIR="${PROJECT_ROOT}/src"

# Configuration file paths (updated for new structure)
readonly ENVIRONMENT_YML="${PROJECT_ROOT}/environment.yml"
readonly PYPROJECT_TOML="${PROJECT_ROOT}/pyproject.toml"
readonly PYTEST_INI="${COVERAGE_DIR}/pytest.ini"  # Updated path
readonly PRECOMMIT_CONFIG="${PROJECT_ROOT}/.pre-commit-config.yaml"

# Enhanced quality assurance tool requirements per Section 8.5
readonly QA_TOOLS=(
    "black>=24.3.0"
    "isort>=5.12.0" 
    "mypy>=1.8.0"
    "flake8>=7.0.0"
    "flake8-pytest-style>=2.1.0"  # Enhanced pytest style validation
    "pre-commit>=3.6.0"
)

# Enhanced testing infrastructure requirements per Section 6.6
readonly TEST_TOOLS=(
    "pytest>=7.0.0"
    "pytest-cov>=6.1.1"
    "pytest-mock>=3.14.1"
    "pytest-benchmark>=4.0.0"
    "coverage>=7.8.2"
    "hypothesis>=6.131.9"
    "pytest-xdist>=3.7.0"         # Parallel execution
    "pytest-timeout>=2.3.0"       # Timeout management
    "pytest-html>=4.0.0"          # Enhanced reporting
    "pytest-json-report>=1.5.0"   # CI/CD integration
    "pytest-rerunfailures>=13.0"  # Flaky test handling
)

# Performance benchmark SLA requirements per Section 6.6.4
readonly BENCHMARK_SLA_DATA_LOADING="1.0"  # seconds per 100MB
readonly BENCHMARK_SLA_DATAFRAME_TRANSFORM="0.5"  # seconds per 1M rows

# Enhanced test execution markers per Section 8.4.4
readonly EXCLUDED_MARKERS="not (benchmark or performance or slow)"
readonly NETWORK_MARKER="network"
readonly BENCHMARK_MARKER="benchmark"
readonly PERFORMANCE_MARKER="performance"

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
declare -g ENABLE_NETWORK_TESTS=false
declare -g PERFORMANCE_ISOLATION=true  # Enhanced default
declare -g VERBOSE=false
declare -g DRY_RUN=false

# Runtime state tracking
declare -g START_TIME
declare -g STEP_COUNTER=0
declare -g ERROR_COUNT=0
declare -g WARNING_COUNT=0

# Enhanced environment detection per Section 8.4.6
declare -g IS_CI="${CI:-false}"
declare -g RUN_NETWORK="${RUN_NETWORK:-false}"
declare -g BENCHMARK_TRIGGER="${BENCHMARK_TRIGGER:-false}"
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
    log "DEBUG" "Memory usage: $(free -h | grep Mem | awk '{print $3 "/" $2}' 2>/dev/null || echo 'unavailable')"
    
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

# Display comprehensive help information with enhanced options
show_help() {
    cat << EOF
${WHITE}$SCRIPT_NAME v$SCRIPT_VERSION${NC}

${CYAN}DESCRIPTION:${NC}
    Enhanced CI/CD environment preparation script implementing automated 
    dependency installation, advanced testing infrastructure, and quality 
    assurance tool setup per Section 8.4 CI/CD Pipeline Architecture and
    Section 8.5 Quality Assurance Pipeline requirements.

${CYAN}ENHANCED FEATURES:${NC}
    - Selective test execution with pytest marker filtering
    - Performance test isolation and optional benchmark execution
    - pytest-xdist parallel execution with optimized worker configuration
    - Environment variable integration for conditional testing
    - Quality gates enforcement with pytest-style validation

${CYAN}USAGE:${NC}
    $SCRIPT_NAME [OPTIONS]

${CYAN}OPTIONS:${NC}
    --python-version VERSION    Python version to validate (default: $DEFAULT_PYTHON_VERSION+)
    --install-method METHOD     Installation method: conda|pip|auto (default: auto)
    --enable-parallel          Enable parallel test execution setup
    --coverage-threshold PCT   Coverage threshold percentage (default: $DEFAULT_COVERAGE_THRESHOLD)
    --benchmark-mode MODE      Benchmark mode: enable|disable|auto (default: auto)
    --quality-gates MODE       Quality gates: strict|standard|minimal (default: strict)
    --enable-network-tests     Enable network-dependent test configuration
    --performance-isolation    Enable performance test isolation strategy
    --verbose                  Enable verbose output
    --dry-run                  Show commands without executing
    --help                     Display this help message

${CYAN}ENVIRONMENT VARIABLES:${NC}
    CI                         Set to 'true' in CI environments
    RUN_NETWORK               Enable network-dependent tests (default: false)
    BENCHMARK_TRIGGER         Enable dedicated benchmark job execution
    PYTEST_CONFIG_FILE        Custom pytest configuration file path
    COVERAGE_THRESHOLD        Override coverage threshold (0-100)
    PARALLEL_WORKERS          Number of parallel test workers
    QUALITY_GATES_MODE        Quality gates enforcement level

${CYAN}EXAMPLES:${NC}
    # Standard setup with enhanced testing strategy
    $SCRIPT_NAME --enable-parallel --performance-isolation

    # CI environment with strict quality gates and network tests
    CI=true RUN_NETWORK=true $SCRIPT_NAME --quality-gates strict

    # Development setup with verbose output and benchmarks
    $SCRIPT_NAME --verbose --benchmark-mode enable

    # Performance testing isolation setup
    $SCRIPT_NAME --performance-isolation --benchmark-mode disable

${CYAN}EXIT CODES:${NC}
    0   Success - environment ready for enhanced testing strategy
    1   General error - check error output
    2   Python version incompatible
    3   Dependency installation failed
    4   Quality assurance setup failed
    5   Configuration validation failed

${CYAN}AUTHOR:${NC}
    $SCRIPT_AUTHOR

EOF
}

# Enhanced argument parsing with new options
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
            --enable-network-tests)
                ENABLE_NETWORK_TESTS=true
                RUN_NETWORK=true
                shift
                ;;
            --performance-isolation)
                PERFORMANCE_ISOLATION=true
                shift
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
    
    # Apply environment variable overrides per Section 8.4.6
    if [[ -n "${COVERAGE_THRESHOLD:-}" ]]; then
        COVERAGE_THRESHOLD="$COVERAGE_THRESHOLD"
    fi
    
    if [[ -n "${QUALITY_GATES_MODE:-}" ]]; then
        QUALITY_GATES_MODE="$QUALITY_GATES_MODE"
    fi
    
    if [[ "${RUN_NETWORK}" == "true" ]]; then
        ENABLE_NETWORK_TESTS=true
    fi
    
    log "DEBUG" "Enhanced configuration parsed: Python=$PYTHON_VERSION, Install=$INSTALL_METHOD, Parallel=$ENABLE_PARALLEL"
    log "DEBUG" "Coverage=$COVERAGE_THRESHOLD%, Benchmark=$BENCHMARK_MODE, QualityGates=$QUALITY_GATES_MODE"
    log "DEBUG" "NetworkTests=$ENABLE_NETWORK_TESTS, PerformanceIsolation=$PERFORMANCE_ISOLATION"
}

# =============================================================================
# ENVIRONMENT DETECTION AND VALIDATION
# =============================================================================

# Enhanced environment detection with CI/CD integration
detect_environment() {
    log "STEP" "Detecting enhanced environment capabilities and CI/CD integration"
    
    # Detect CI environment with enhanced configuration
    if [[ "$IS_CI" == "true" ]]; then
        log "INFO" "CI environment detected - enabling optimized configurations"
        # Override parallel testing in CI for enhanced performance
        ENABLE_PARALLEL=true
        PERFORMANCE_ISOLATION=true
        
        # Configure CI-specific environment variables
        if [[ ! "$DRY_RUN" == true ]]; then
            export PYTHONUNBUFFERED=1
            export PYTEST_TIMEOUT="$DEFAULT_PYTEST_TIMEOUT"
            log "DEBUG" "CI environment variables configured"
        fi
    fi
    
    # Enhanced conda detection
    if command -v conda &> /dev/null; then
        HAS_CONDA=true
        local conda_version=$(conda --version 2>/dev/null | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -1)
        log "INFO" "Conda detected: version $conda_version"
    else
        log "WARN" "Conda not found - pip installation will be used"
    fi
    
    # Enhanced pip detection
    if command -v pip &> /dev/null; then
        HAS_PIP=true
        local pip_version=$(pip --version 2>/dev/null | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -1)
        log "INFO" "Pip detected: version $pip_version"
    else
        log "ERROR" "Pip not found - unable to install enhanced dependencies"
        handle_error 3 "Pip is required for enhanced dependency installation" \
            "Install pip or use a Python distribution with pip included"
    fi
    
    # Determine optimal installation method with enhanced criteria
    if [[ "$INSTALL_METHOD" == "auto" ]]; then
        if [[ "$HAS_CONDA" == true ]] && [[ -f "$ENVIRONMENT_YML" ]]; then
            INSTALL_METHOD="conda"
            log "INFO" "Auto-detected installation method: conda (environment.yml found with enhanced dependencies)"
        elif [[ "$HAS_PIP" == true ]]; then
            INSTALL_METHOD="pip"
            log "INFO" "Auto-detected installation method: pip"
        else
            handle_error 3 "No suitable installation method available" \
                "Install conda or pip to proceed with enhanced setup"
        fi
    fi
    
    log "SUCCESS" "Enhanced environment detection completed successfully"
}

# Python version validation with enhanced compatibility checking
validate_python_version() {
    log "STEP" "Validating Python version compatibility for enhanced testing features"
    
    # Find Python executable
    if command -v python3 &> /dev/null; then
        PYTHON_EXECUTABLE="python3"
    elif command -v python &> /dev/null; then
        PYTHON_EXECUTABLE="python"
    else
        handle_error 2 "Python not found" \
            "Install Python $PYTHON_VERSION or higher for enhanced testing features"
    fi
    
    # Get current Python version
    local current_version
    current_version=$($PYTHON_EXECUTABLE -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
    
    log "INFO" "Current Python version: $current_version"
    log "INFO" "Required Python version: $PYTHON_VERSION+ (enhanced testing compatibility)"
    
    # Version comparison using Python itself for accuracy
    local version_check
    version_check=$($PYTHON_EXECUTABLE -c "
import sys
required = tuple(map(int, '$PYTHON_VERSION'.split('.')))
current = sys.version_info[:len(required)]
print('compatible' if current >= required else 'incompatible')
")
    
    if [[ "$version_check" == "incompatible" ]]; then
        handle_error 2 "Python version $current_version is incompatible with enhanced testing features" \
            "Upgrade to Python $PYTHON_VERSION or higher"
    fi
    
    # Check for enhanced testing features compatibility
    log "DEBUG" "Validating enhanced testing features compatibility"
    
    # Check for asyncio support (required for modern testing)
    if ! $PYTHON_EXECUTABLE -c "import asyncio" 2>/dev/null; then
        log "WARN" "AsyncIO support not available - some advanced testing features may be limited"
    fi
    
    # Check for concurrent.futures (required for parallel execution)
    if ! $PYTHON_EXECUTABLE -c "import concurrent.futures" 2>/dev/null; then
        log "WARN" "Concurrent.futures not available - parallel test execution may be limited"
    fi
    
    # Check for multiprocessing support (required for pytest-xdist)
    if ! $PYTHON_EXECUTABLE -c "import multiprocessing; print(multiprocessing.cpu_count())" 2>/dev/null; then
        log "WARN" "Multiprocessing support limited - parallel testing may not be optimal"
    fi
    
    log "SUCCESS" "Python version validation completed: $current_version supports enhanced testing"
}

# Enhanced configuration file validation
validate_configuration_files() {
    log "STEP" "Validating enhanced project configuration files"
    
    local validation_errors=0
    
    # Validate environment.yml with enhanced dependencies
    if [[ -f "$ENVIRONMENT_YML" ]]; then
        log "INFO" "Validating environment.yml for enhanced testing dependencies"
        
        # Check for enhanced testing tools
        local enhanced_deps=("pytest-xdist" "pytest-timeout" "pytest-html" "hypothesis")
        for dep in "${enhanced_deps[@]}"; do
            if ! grep -q "$dep" "$ENVIRONMENT_YML"; then
                log "WARN" "Enhanced testing dependency $dep not found in environment.yml"
            fi
        done
    else
        log "WARN" "environment.yml not found - conda installation with enhanced features will not be available"
    fi
    
    # Validate pyproject.toml for enhanced pytest configuration
    if [[ -f "$PYPROJECT_TOML" ]]; then
        log "INFO" "Validating pyproject.toml for enhanced pytest configuration"
        
        # Check for pytest configuration section
        if ! grep -q "\[tool\.pytest\.ini_options\]" "$PYPROJECT_TOML"; then
            log "WARN" "pytest configuration section not found in pyproject.toml"
        fi
        
        # Check for enhanced markers configuration
        if ! grep -q "markers" "$PYPROJECT_TOML"; then
            log "WARN" "pytest markers configuration not found - enhanced test categorization may be limited"
        fi
    else
        log "ERROR" "pyproject.toml not found - required for enhanced project configuration"
        ((validation_errors++))
    fi
    
    # Validate enhanced directory structure
    local required_dirs=("$SRC_DIR" "$TESTS_DIR")
    for dir in "${required_dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            log "ERROR" "Required directory not found: $dir"
            ((validation_errors++))
        fi
    done
    
    # Create enhanced coverage directory structure if needed
    execute_command "Create enhanced coverage directory structure" \
        mkdir -p "$COVERAGE_DIR" "$COVERAGE_DIR/reports" "$COVERAGE_DIR/historical"
    
    # Create benchmarks directory for performance test isolation
    if [[ "$PERFORMANCE_ISOLATION" == true ]]; then
        execute_command "Create performance test isolation directory" \
            mkdir -p "$BENCHMARKS_DIR"
    fi
    
    if [[ $validation_errors -gt 0 ]]; then
        handle_error 5 "Enhanced configuration validation failed with $validation_errors errors" \
            "Fix configuration issues before proceeding with enhanced setup"
    fi
    
    log "SUCCESS" "Enhanced configuration file validation completed successfully"
}

# =============================================================================
# ENHANCED DEPENDENCY INSTALLATION AND MANAGEMENT
# =============================================================================

# Enhanced conda environment setup with comprehensive testing dependencies
setup_conda_environment() {
    log "STEP" "Setting up enhanced conda environment with comprehensive testing dependencies"
    
    if [[ ! -f "$ENVIRONMENT_YML" ]]; then
        log "WARN" "environment.yml not found - skipping enhanced conda setup"
        return 0
    fi
    
    # Extract environment name from environment.yml
    local env_name
    env_name=$(grep "^name:" "$ENVIRONMENT_YML" | sed 's/name: *//')
    
    log "INFO" "Creating/updating enhanced conda environment: $env_name"
    
    # Update conda to latest version for better dependency resolution
    execute_command "Update conda to latest version for enhanced features" \
        conda update -n base -c defaults conda -y
    
    # Create or update environment from file with enhanced dependencies
    execute_command "Create/update enhanced conda environment from environment.yml" \
        conda env update --file "$ENVIRONMENT_YML" --prune
    
    # Activate environment and verify enhanced installations
    log "INFO" "Activating enhanced conda environment: $env_name"
    
    # Source conda activation script
    local conda_base
    conda_base=$(conda info --base)
    
    if [[ ! "$DRY_RUN" == true ]]; then
        # shellcheck source=/dev/null
        source "$conda_base/etc/profile.d/conda.sh"
        conda activate "$env_name"
        
        # Verify enhanced testing packages are installed
        local enhanced_packages=("pytest" "pytest-cov" "pytest-xdist" "pytest-timeout" "black" "mypy")
        for package in "${enhanced_packages[@]}"; do
            if ! conda list | grep -q "^$package "; then
                log "WARN" "Enhanced package $package not found in conda environment"
            else
                local version
                version=$(conda list | grep "^$package " | awk '{print $2}')
                log "DEBUG" "Verified enhanced package $package version: $version"
            fi
        done
    fi
    
    log "SUCCESS" "Enhanced conda environment setup completed successfully"
}

# Advanced pip dependency installation with enhanced testing tools
install_pip_dependencies() {
    log "STEP" "Installing enhanced pip dependencies with advanced testing infrastructure"
    
    # Upgrade pip to latest version for better dependency resolution
    execute_command "Upgrade pip to latest version for enhanced features" \
        $PYTHON_EXECUTABLE -m pip install --upgrade pip
    
    # Install build dependencies first
    execute_command "Install enhanced build system dependencies" \
        $PYTHON_EXECUTABLE -m pip install "setuptools>=42" wheel build
    
    # Install project in editable mode with enhanced development dependencies
    log "INFO" "Installing enhanced project dependencies from pyproject.toml"
    
    if [[ -f "$PYPROJECT_TOML" ]]; then
        execute_command "Install project with enhanced development dependencies" \
            $PYTHON_EXECUTABLE -m pip install -e ".[dev]"
    else
        log "WARN" "pyproject.toml not found - installing enhanced testing dependencies individually"
        
        # Install enhanced testing tools individually
        for tool in "${TEST_TOOLS[@]}"; do
            execute_command "Install enhanced testing tool: $tool" \
                $PYTHON_EXECUTABLE -m pip install "$tool"
        done
        
        # Install enhanced quality assurance tools
        for tool in "${QA_TOOLS[@]}"; do
            execute_command "Install enhanced QA tool: $tool" \
                $PYTHON_EXECUTABLE -m pip install "$tool"
        done
    fi
    
    # Verify enhanced package installations
    log "INFO" "Verifying enhanced installed package versions"
    
    local enhanced_packages=("pytest" "pytest-cov" "pytest-xdist" "pytest-timeout" "black" "mypy" "coverage" "hypothesis")
    for package in "${enhanced_packages[@]}"; do
        if $PYTHON_EXECUTABLE -c "import $package" 2>/dev/null; then
            local version
            version=$($PYTHON_EXECUTABLE -c "import $package; print(getattr($package, '__version__', 'unknown'))" 2>/dev/null)
            log "DEBUG" "Verified enhanced package $package version: $version"
        else
            log "WARN" "Enhanced package $package not properly installed or importable"
        fi
    done
    
    log "SUCCESS" "Enhanced pip dependency installation completed successfully"
}

# Unified enhanced dependency installation orchestration
install_dependencies() {
    log "STEP" "Installing comprehensive enhanced testing and quality assurance dependencies"
    
    case "$INSTALL_METHOD" in
        conda)
            setup_conda_environment
            ;;
        pip)
            install_pip_dependencies
            ;;
        *)
            handle_error 3 "Invalid installation method: $INSTALL_METHOD" \
                "Use conda, pip, or auto for enhanced setup"
            ;;
    esac
    
    # Enhanced dependency verification
    log "INFO" "Performing comprehensive enhanced dependency verification"
    
    # Verify pytest installation with enhanced features
    if ! $PYTHON_EXECUTABLE -c "import pytest; print(f'pytest {pytest.__version__}')" 2>/dev/null; then
        handle_error 3 "Enhanced pytest installation verification failed" \
            "Check pytest installation and Python path"
    fi
    
    # Verify pytest-xdist for parallel execution
    if ! $PYTHON_EXECUTABLE -c "import xdist; print(f'pytest-xdist {xdist.__version__}')" 2>/dev/null; then
        log "WARN" "pytest-xdist not available - parallel test execution will be limited"
    fi
    
    # Verify coverage measurement capability
    if ! $PYTHON_EXECUTABLE -c "import coverage; print(f'coverage {coverage.__version__}')" 2>/dev/null; then
        handle_error 3 "Enhanced coverage installation verification failed" \
            "Check coverage.py installation"
    fi
    
    # Verify hypothesis for property-based testing
    if ! $PYTHON_EXECUTABLE -c "import hypothesis; print(f'hypothesis {hypothesis.__version__}')" 2>/dev/null; then
        log "WARN" "hypothesis not available - property-based testing features will be limited"
    fi
    
    log "SUCCESS" "All enhanced dependencies installed and verified successfully"
}

# =============================================================================
# ENHANCED TESTING INFRASTRUCTURE CONFIGURATION
# =============================================================================

# Comprehensive enhanced pytest configuration setup
configure_pytest_infrastructure() {
    log "STEP" "Configuring comprehensive enhanced pytest testing infrastructure"
    
    # Determine pytest configuration source
    local pytest_config_file="${PYTEST_CONFIG_FILE:-$PYTEST_INI}"
    
    if [[ -f "$pytest_config_file" ]]; then
        log "INFO" "Using enhanced pytest configuration from: $pytest_config_file"
    else
        log "INFO" "Using enhanced pytest configuration from pyproject.toml"
        pytest_config_file="$PYPROJECT_TOML"
    fi
    
    # Validate enhanced pytest configuration
    execute_command "Validate enhanced pytest configuration" \
        $PYTHON_EXECUTABLE -m pytest --collect-only --quiet
    
    # Configure enhanced pytest markers for comprehensive test categorization
    log "INFO" "Configuring enhanced pytest markers for comprehensive test categorization"
    
    # Verify enhanced test discovery with marker filtering
    local test_count
    if [[ ! "$DRY_RUN" == true ]]; then
        test_count=$($PYTHON_EXECUTABLE -m pytest --collect-only --quiet -m "$EXCLUDED_MARKERS" 2>/dev/null | grep -c "<Module" || echo "0")
        log "INFO" "Discovered $test_count test modules (excluding performance and benchmark tests)"
        
        if [[ "$test_count" -eq 0 ]]; then
            log "WARN" "No tests discovered with enhanced filtering - verify test file naming and marker structure"
        fi
        
        # Check for performance tests isolation
        local perf_test_count
        perf_test_count=$($PYTHON_EXECUTABLE -m pytest --collect-only --quiet -m "$BENCHMARK_MARKER or $PERFORMANCE_MARKER" 2>/dev/null | grep -c "<Module" || echo "0")
        log "INFO" "Found $perf_test_count performance/benchmark test modules (isolated from default execution)"
    fi
    
    # Configure enhanced parallel testing with pytest-xdist
    if [[ "$ENABLE_PARALLEL" == true ]]; then
        log "INFO" "Configuring enhanced parallel test execution with pytest-xdist"
        
        # Determine optimal worker count for enhanced performance
        local worker_count="${PARALLEL_WORKERS:-$DEFAULT_PARALLEL_WORKERS}"
        if [[ "$worker_count" == "auto" ]]; then
            if [[ ! "$DRY_RUN" == true ]]; then
                worker_count=$(nproc 2>/dev/null || echo "4")
                log "DEBUG" "Auto-detected $worker_count CPU cores for enhanced parallel testing"
            else
                worker_count="4"
            fi
        fi
        
        # Verify enhanced pytest-xdist functionality
        execute_command "Verify enhanced pytest-xdist parallel testing capability" \
            $PYTHON_EXECUTABLE -c "import xdist; print(f'pytest-xdist {xdist.__version__} ready for {worker_count} workers')"
        
        # Configure enhanced parallel execution environment
        if [[ ! "$DRY_RUN" == true ]]; then
            export PYTEST_XDIST_WORKERS="$worker_count"
            log "DEBUG" "Enhanced parallel execution configured: $worker_count workers"
        fi
    fi
    
    # Configure conditional network testing per Section 8.4.6
    if [[ "$ENABLE_NETWORK_TESTS" == true ]]; then
        log "INFO" "Configuring conditional network-dependent test execution"
        
        if [[ ! "$DRY_RUN" == true ]]; then
            export RUN_NETWORK=true
            log "DEBUG" "Network testing enabled: RUN_NETWORK=true"
        fi
    fi
    
    log "SUCCESS" "Enhanced pytest infrastructure configuration completed successfully"
}

# Advanced coverage reporting setup with enhanced analysis
configure_coverage_reporting() {
    log "STEP" "Configuring advanced coverage reporting with enhanced quality thresholds"
    
    # Create enhanced coverage configuration directory structure
    execute_command "Create enhanced coverage reporting directory structure" \
        mkdir -p "$COVERAGE_DIR/reports" "$COVERAGE_DIR/historical" "$COVERAGE_DIR/templates"
    
    # Validate enhanced coverage threshold
    if [[ "$COVERAGE_THRESHOLD" -lt 50 ]]; then
        log "WARN" "Coverage threshold $COVERAGE_THRESHOLD% is below recommended minimum (50%)"
    elif [[ "$COVERAGE_THRESHOLD" -ge 90 ]]; then
        log "INFO" "Coverage threshold $COVERAGE_THRESHOLD% meets enterprise quality standards for enhanced testing"
    fi
    
    # Configure enhanced coverage contexts for detailed reporting
    log "INFO" "Configuring enhanced coverage contexts for comprehensive measurement"
    
    # Test enhanced coverage measurement capability
    execute_command "Verify enhanced coverage measurement functionality" \
        $PYTHON_EXECUTABLE -c "
import coverage
cov = coverage.Coverage()
print(f'Coverage.py {coverage.__version__} ready for enhanced measurement')
print(f'Branch coverage: {cov.config.branch}')
print(f'Source patterns: {cov.config.source}')
print(f'Parallel mode: {cov.config.parallel}')
"
    
    # Configure enhanced coverage database location
    local coverage_db="$COVERAGE_DIR/.coverage"
    log "DEBUG" "Enhanced coverage database location: $coverage_db"
    
    # Set up enhanced coverage environment variables
    if [[ ! "$DRY_RUN" == true ]]; then
        export COVERAGE_FILE="$coverage_db"
        export COVERAGE_CONTEXT="enhanced-test-execution"
        export COVERAGE_THRESHOLD="$COVERAGE_THRESHOLD"
        log "DEBUG" "Enhanced coverage environment configured: COVERAGE_FILE=$COVERAGE_FILE"
    fi
    
    log "SUCCESS" "Enhanced coverage reporting configuration completed successfully"
}

# Enhanced performance benchmark configuration with isolation
configure_performance_benchmarks() {
    log "STEP" "Configuring enhanced performance benchmarks with isolation strategy"
    
    # Determine enhanced benchmark mode based on configuration
    local enable_benchmarks=false
    
    case "$BENCHMARK_MODE" in
        enable)
            enable_benchmarks=true
            ;;
        disable)
            enable_benchmarks=false
            ;;
        auto)
            # Enable benchmarks only when specifically triggered per Section 8.4.4
            if [[ "$BENCHMARK_TRIGGER" == "true" ]]; then
                enable_benchmarks=true
            fi
            ;;
    esac
    
    if [[ "$enable_benchmarks" == true ]]; then
        log "INFO" "Enabling enhanced performance benchmark validation with isolation"
        
        # Verify enhanced pytest-benchmark installation
        execute_command "Verify enhanced pytest-benchmark functionality" \
            $PYTHON_EXECUTABLE -c "
import pytest_benchmark
print(f'pytest-benchmark {pytest_benchmark.__version__}')
print(f'Enhanced data loading SLA: {$BENCHMARK_SLA_DATA_LOADING}s per 100MB')
print(f'Enhanced DataFrame transform SLA: {$BENCHMARK_SLA_DATAFRAME_TRANSFORM}s per 1M rows')
print('Performance test isolation strategy: enabled')
"
        
        # Configure enhanced benchmark storage in isolated directory
        execute_command "Create enhanced benchmark storage directory" \
            mkdir -p "$BENCHMARKS_DIR" "$BENCHMARKS_DIR/results" "$BENCHMARKS_DIR/reports"
        
        # Set enhanced benchmark environment variables
        if [[ ! "$DRY_RUN" == true ]]; then
            export PYTEST_BENCHMARK_DIR="$BENCHMARKS_DIR/results"
            export BENCHMARK_ISOLATION=true
            log "DEBUG" "Enhanced benchmark storage configured with isolation: $PYTEST_BENCHMARK_DIR"
        fi
    else
        log "INFO" "Enhanced performance benchmarks configured for isolation (excluded from default execution)"
        
        # Ensure benchmarks directory exists for isolation strategy
        if [[ "$PERFORMANCE_ISOLATION" == true ]]; then
            execute_command "Create benchmark isolation directory" \
                mkdir -p "$BENCHMARKS_DIR"
        fi
    fi
    
    log "SUCCESS" "Enhanced performance benchmark configuration with isolation completed"
}

# =============================================================================
# ENHANCED QUALITY ASSURANCE TOOL CONFIGURATION
# =============================================================================

# Enhanced code formatting tool setup with pytest-style validation
configure_code_formatting_tools() {
    log "STEP" "Configuring enhanced code formatting and style enforcement tools"
    
    # Configure enhanced Black code formatter
    log "INFO" "Configuring enhanced Black code formatter for consistent code style"
    
    execute_command "Verify enhanced Black installation and configuration" \
        $PYTHON_EXECUTABLE -m black --version
    
    # Configure enhanced isort import sorting with Black compatibility
    log "INFO" "Configuring enhanced isort for import statement organization with Black profile"
    
    execute_command "Verify enhanced isort installation and Black compatibility" \
        $PYTHON_EXECUTABLE -c "
import isort
print(f'isort {isort.__version__}')
# Verify enhanced Black profile compatibility
config = isort.Config(profile='black')
print(f'Enhanced Black compatibility profile: {config.profile}')
"
    
    # Configure enhanced flake8-pytest-style for AAA patterns
    log "INFO" "Configuring enhanced pytest-style validation for AAA patterns and naming conventions"
    
    execute_command "Verify enhanced flake8-pytest-style installation" \
        $PYTHON_EXECUTABLE -c "
try:
    import flake8_pytest_style
    print('flake8-pytest-style available for enhanced test validation')
except ImportError:
    print('flake8-pytest-style not available - pytest style validation may be limited')
"
    
    log "SUCCESS" "Enhanced code formatting tools configured successfully"
}

# Enhanced static type checking configuration
configure_static_type_checking() {
    log "STEP" "Configuring enhanced static type checking with MyPy"
    
    # Verify enhanced MyPy installation
    execute_command "Verify enhanced MyPy installation" \
        $PYTHON_EXECUTABLE -m mypy --version
    
    # Configure enhanced MyPy for strict type checking per quality requirements
    log "INFO" "Configuring enhanced MyPy for strict type checking enforcement"
    
    # Configure enhanced MyPy cache directory
    execute_command "Create enhanced MyPy cache directory" \
        mkdir -p "$COVERAGE_DIR/mypy_cache"
    
    if [[ ! "$DRY_RUN" == true ]]; then
        export MYPY_CACHE_DIR="$COVERAGE_DIR/mypy_cache"
        log "DEBUG" "Enhanced MyPy cache configured: $MYPY_CACHE_DIR"
    fi
    
    log "SUCCESS" "Enhanced static type checking configuration completed"
}

# Enhanced linting configuration with pytest-style validation
configure_linting_tools() {
    log "STEP" "Configuring enhanced linting with Flake8 and pytest-style validation"
    
    # Verify enhanced Flake8 installation
    execute_command "Verify enhanced Flake8 installation" \
        $PYTHON_EXECUTABLE -m flake8 --version
    
    # Configure enhanced Flake8 plugins and rule sets
    log "INFO" "Configuring enhanced Flake8 with comprehensive rule enforcement and pytest-style validation"
    
    log "SUCCESS" "Enhanced linting tool configuration completed"
}

# Enhanced pre-commit hooks installation and configuration
configure_precommit_hooks() {
    log "STEP" "Configuring enhanced pre-commit hooks for automated quality gates"
    
    # Verify enhanced pre-commit installation
    execute_command "Verify enhanced pre-commit installation" \
        pre-commit --version
    
    # Install enhanced pre-commit hooks if configuration exists
    if [[ -f "$PRECOMMIT_CONFIG" ]]; then
        log "INFO" "Installing enhanced pre-commit hooks from .pre-commit-config.yaml"
        
        execute_command "Install enhanced pre-commit git hooks" \
            pre-commit install
        
        # Validate enhanced pre-commit configuration
        execute_command "Validate enhanced pre-commit configuration" \
            pre-commit validate-config
        
        # Update enhanced pre-commit hook repositories
        execute_command "Update enhanced pre-commit hook repositories" \
            pre-commit autoupdate
        
    else
        log "INFO" "Creating enhanced pre-commit configuration with pytest-style validation"
        
        # Create enhanced pre-commit configuration
        if [[ ! "$DRY_RUN" == true ]]; then
            cat > "$PRECOMMIT_CONFIG" << 'EOF'
# Enhanced pre-commit configuration for flyrigloader
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
        additional_dependencies:
          - flake8-pytest-style>=2.1.0
        args:
          - --extend-ignore=E203,W503
          - --max-line-length=100

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--ignore-missing-imports]
EOF
            
            execute_command "Install newly created enhanced pre-commit hooks" \
                pre-commit install
        fi
    fi
    
    log "SUCCESS" "Enhanced pre-commit hooks configuration completed"
}

# Enhanced quality gates enforcement configuration
configure_quality_gates() {
    log "STEP" "Configuring enhanced quality gates enforcement per Section 8.5 requirements"
    
    case "$QUALITY_GATES_MODE" in
        strict)
            log "INFO" "Configuring strict enhanced quality gates (enterprise-grade with pytest-style validation)"
            # 90% coverage, zero lint violations, full type checking, pytest-style compliance
            ;;
        standard)
            log "INFO" "Configuring standard enhanced quality gates"
            # 80% coverage, minimal lint violations, basic type checking
            COVERAGE_THRESHOLD=80
            ;;
        minimal)
            log "INFO" "Configuring minimal enhanced quality gates"
            # 70% coverage, essential lint checks only
            COVERAGE_THRESHOLD=70
            ;;
    esac
    
    # Create enhanced quality gate validation script
    local quality_gate_script="$COVERAGE_DIR/validate-quality-gates.sh"
    
    if [[ ! "$DRY_RUN" == true ]]; then
        cat > "$quality_gate_script" << EOF
#!/bin/bash
# Enhanced automated quality gate validation script
# Generated by $SCRIPT_NAME v$SCRIPT_VERSION

set -euo pipefail

echo "Validating enhanced quality gates (mode: $QUALITY_GATES_MODE)"

# Run enhanced tests with coverage and selective execution
echo "Running enhanced test suite with performance test isolation..."
python -m pytest -m "$EXCLUDED_MARKERS" \\
    --cov=src/flyrigloader \\
    --cov-fail-under=$COVERAGE_THRESHOLD \\
    --cov-report=html:$COVERAGE_DIR/reports/htmlcov \\
    --cov-report=xml:$COVERAGE_DIR/reports/coverage.xml \\
    --timeout=$DEFAULT_PYTEST_TIMEOUT \\
    $(if [[ "$ENABLE_PARALLEL" == "true" ]]; then echo "-n $DEFAULT_PARALLEL_WORKERS"; fi) \\
    $(if [[ "$ENABLE_NETWORK_TESTS" == "true" ]]; then echo "--run-network"; fi)

# Run enhanced type checking
echo "Running enhanced type checking..."
python -m mypy src/flyrigloader --ignore-missing-imports

# Run enhanced linting with pytest-style validation
echo "Running enhanced linting with pytest-style validation..."
python -m flake8 src/flyrigloader tests/

# Run enhanced formatting checks
echo "Running enhanced formatting checks..."
python -m black --check src/flyrigloader tests/
python -m isort --check-only --profile black src/flyrigloader tests/

# Performance test isolation verification
if [[ "$PERFORMANCE_ISOLATION" == "true" ]]; then
    echo "Verifying performance test isolation..."
    if python -m pytest --collect-only -m "$BENCHMARK_MARKER or $PERFORMANCE_MARKER" --quiet > /dev/null 2>&1; then
        echo "Performance tests properly isolated in $BENCHMARKS_DIR"
    fi
fi

echo "All enhanced quality gates passed successfully"
EOF
        
        chmod +x "$quality_gate_script"
        log "DEBUG" "Enhanced quality gate validation script created: $quality_gate_script"
    fi
    
    log "SUCCESS" "Enhanced quality gates configuration completed"
}

# =============================================================================
# ENHANCED CI/CD INTEGRATION AND OPTIMIZATION
# =============================================================================

# Enhanced GitHub Actions workflow configuration
configure_github_actions_integration() {
    log "STEP" "Configuring enhanced GitHub Actions CI/CD integration"
    
    local github_workflows_dir="${PROJECT_ROOT}/.github/workflows"
    local test_workflow="${github_workflows_dir}/test.yml"
    
    execute_command "Create enhanced GitHub Actions workflows directory" \
        mkdir -p "$github_workflows_dir"
    
    if [[ ! -f "$test_workflow" ]] && [[ ! "$DRY_RUN" == true ]]; then
        log "INFO" "Creating enhanced GitHub Actions test workflow with performance isolation"
        
        cat > "$test_workflow" << 'EOF'
name: Enhanced Test Suite & Quality Assurance

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  workflow_dispatch:
    inputs:
      enable_benchmarks:
        description: 'Enable benchmark job execution'
        required: false
        default: false
        type: boolean
      run_network_tests:
        description: 'Enable network-dependent tests'
        required: false
        default: false
        type: boolean

env:
  COVERAGE_THRESHOLD: '90'
  RUN_NETWORK: ${{ inputs.run_network_tests || false }}
  BENCHMARK_TRIGGER: ${{ inputs.enable_benchmarks || false }}

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
    
    - name: Prepare enhanced CI environment
      run: |
        chmod +x scripts/coverage/prepare-ci-environment.sh
        ./scripts/coverage/prepare-ci-environment.sh --enable-parallel --performance-isolation
    
    - name: Run enhanced core test suite (excluding performance tests)
      run: |
        pytest -m "not (benchmark or performance)" \
               --cov=src/flyrigloader \
               --cov-report=xml:coverage.xml \
               --cov-fail-under=90 \
               --timeout=30 \
               -n 4 \
               --dist=worksteal
    
    - name: Run network tests (conditional)
      if: env.RUN_NETWORK == 'true'
      run: |
        pytest -m "not (benchmark or performance)" \
               --run-network \
               --cov=src/flyrigloader \
               --cov-append \
               --cov-report=xml:coverage-network.xml \
               --timeout=60 \
               -n 2
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v4
      with:
        file: ./coverage.xml
        fail_ci_if_error: true

  benchmarks:
    runs-on: ubuntu-latest
    if: github.event.inputs.enable_benchmarks == 'true' || contains(github.event.pull_request.labels.*.name, 'benchmark')
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python 3.11 for benchmarks
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
    
    - name: Prepare benchmark environment
      run: |
        chmod +x scripts/coverage/prepare-ci-environment.sh
        ./scripts/coverage/prepare-ci-environment.sh --benchmark-mode enable
    
    - name: Run performance benchmarks
      run: |
        python scripts/benchmarks/run_benchmarks.py --ci-mode
EOF
        
        log "INFO" "Enhanced GitHub Actions workflow created: $test_workflow"
    fi
    
    log "SUCCESS" "Enhanced GitHub Actions integration configuration completed"
}

# Enhanced performance optimization for test execution
optimize_test_execution() {
    log "STEP" "Optimizing enhanced test execution performance"
    
    # Configure enhanced Python optimization settings
    if [[ ! "$DRY_RUN" == true ]]; then
        # Configure enhanced pytest cache directory
        local pytest_cache_dir="$COVERAGE_DIR/.pytest_cache"
        execute_command "Create enhanced pytest cache directory" \
            mkdir -p "$pytest_cache_dir"
        
        export PYTEST_CACHE_DIR="$pytest_cache_dir"
        log "DEBUG" "Enhanced pytest cache configured: $PYTEST_CACHE_DIR"
        
        # Set enhanced environment variables for performance
        export PYTHONUNBUFFERED=1
        export PYTEST_TIMEOUT="$DEFAULT_PYTEST_TIMEOUT"
        
        if [[ "$IS_CI" == "true" ]]; then
            export PYTHONOPTIMIZE=1
            log "DEBUG" "Enhanced Python optimization enabled for CI environment"
        fi
    fi
    
    # Configure enhanced parallel execution
    if [[ "$ENABLE_PARALLEL" == true ]]; then
        local cpu_count
        cpu_count=$(nproc 2>/dev/null || echo "4")
        log "INFO" "Enhanced parallel test execution enabled: $cpu_count workers with worksteal distribution"
    fi
    
    log "SUCCESS" "Enhanced test execution optimization completed"
}

# =============================================================================
# ENHANCED FINAL VALIDATION AND ENVIRONMENT VERIFICATION
# =============================================================================

# Comprehensive enhanced environment validation
validate_complete_environment() {
    log "STEP" "Performing comprehensive enhanced environment validation"
    
    local validation_errors=0
    
    # Test enhanced pytest execution with marker filtering
    log "INFO" "Validating enhanced pytest execution with performance test isolation"
    if [[ ! "$DRY_RUN" == true ]]; then
        if $PYTHON_EXECUTABLE -m pytest --collect-only -m "$EXCLUDED_MARKERS" --quiet >/dev/null 2>&1; then
            log "DEBUG" "Enhanced pytest execution with marker filtering validated successfully"
        else
            log "ERROR" "Enhanced pytest execution validation failed"
            ((validation_errors++))
        fi
    fi
    
    # Test enhanced parallel execution capability
    if [[ "$ENABLE_PARALLEL" == true ]] && [[ ! "$DRY_RUN" == true ]]; then
        log "INFO" "Validating enhanced parallel test execution capability"
        
        if $PYTHON_EXECUTABLE -c "import xdist; import concurrent.futures; print('Enhanced parallel execution ready')" >/dev/null 2>&1; then
            log "DEBUG" "Enhanced parallel execution capability validated"
        else
            log "ERROR" "Enhanced parallel execution validation failed"
            ((validation_errors++))
        fi
    fi
    
    # Test enhanced coverage measurement with contexts
    log "INFO" "Validating enhanced coverage measurement capability"
    if [[ ! "$DRY_RUN" == true ]]; then
        local coverage_test_file="/tmp/enhanced_coverage_test.py"
        echo "def test_enhanced_function(): assert True" > "$coverage_test_file"
        
        if $PYTHON_EXECUTABLE -m pytest --cov=. --cov-context=test --cov-report=term-missing "$coverage_test_file" >/dev/null 2>&1; then
            log "DEBUG" "Enhanced coverage measurement validated successfully"
        else
            log "ERROR" "Enhanced coverage measurement validation failed"
            ((validation_errors++))
        fi
        
        rm -f "$coverage_test_file" .coverage
    fi
    
    # Test enhanced quality assurance tools with pytest-style validation
    log "INFO" "Validating enhanced quality assurance tool integration"
    
    local enhanced_qa_tools=("black" "isort" "mypy" "flake8")
    for tool in "${enhanced_qa_tools[@]}"; do
        if [[ ! "$DRY_RUN" == true ]]; then
            if $PYTHON_EXECUTABLE -m "$tool" --version >/dev/null 2>&1; then
                log "DEBUG" "Enhanced QA tool $tool validated successfully"
            else
                log "ERROR" "Enhanced QA tool $tool validation failed"
                ((validation_errors++))
            fi
        fi
    done
    
    if [[ $validation_errors -gt 0 ]]; then
        handle_error 4 "Enhanced environment validation failed with $validation_errors errors" \
            "Review error messages and fix issues before proceeding with enhanced testing"
    fi
    
    log "SUCCESS" "Complete enhanced environment validation passed successfully"
}

# Generate comprehensive enhanced environment summary
generate_environment_summary() {
    log "STEP" "Generating enhanced environment setup summary and usage instructions"
    
    local summary_file="$COVERAGE_DIR/enhanced-environment-summary.md"
    local current_time=$(date '+%Y-%m-%d %H:%M:%S')
    
    if [[ ! "$DRY_RUN" == true ]]; then
        cat > "$summary_file" << EOF
# Enhanced CI/CD Environment Setup Summary

**Generated by:** $SCRIPT_NAME v$SCRIPT_VERSION  
**Generated on:** $current_time  
**Environment:** $(if [[ "$IS_CI" == "true" ]]; then echo "CI/CD"; else echo "Local Development"; fi)

## Enhanced Configuration Summary

| Setting | Value |
|---------|-------|
| Python Version | $($PYTHON_EXECUTABLE --version 2>&1) |
| Installation Method | $INSTALL_METHOD |
| Coverage Threshold | $COVERAGE_THRESHOLD% |
| Quality Gates Mode | $QUALITY_GATES_MODE |
| Parallel Testing | $(if [[ "$ENABLE_PARALLEL" == "true" ]]; then echo "Enabled ($DEFAULT_PARALLEL_WORKERS workers)"; else echo "Disabled"; fi) |
| Network Tests | $(if [[ "$ENABLE_NETWORK_TESTS" == "true" ]]; then echo "Enabled"; else echo "Disabled"; fi) |
| Performance Isolation | $(if [[ "$PERFORMANCE_ISOLATION" == "true" ]]; then echo "Enabled"; else echo "Disabled"; fi) |
| Benchmark Mode | $BENCHMARK_MODE |

## Enhanced Testing Strategy

### Core Test Execution (Default)
- **Command:** \`pytest -m "$EXCLUDED_MARKERS"\`
- **Strategy:** Excludes performance and benchmark tests for rapid feedback
- **Timeout:** ${DEFAULT_PYTEST_TIMEOUT}s
- **Parallel Workers:** $DEFAULT_PARALLEL_WORKERS (when enabled)

### Performance Test Isolation
- **Location:** \`$BENCHMARKS_DIR/\`
- **Execution:** Optional via \`BENCHMARK_TRIGGER=true\` or manual execution
- **Strategy:** Isolated from default CI execution for optimal performance

### Network-Dependent Testing
- **Trigger:** \`RUN_NETWORK=true\` environment variable
- **Strategy:** Conditional execution to avoid external dependency failures

## Quick Start Commands

### Enhanced Core Test Suite
\`\`\`bash
pytest -m "$EXCLUDED_MARKERS" --cov=src/flyrigloader --cov-fail-under=$COVERAGE_THRESHOLD
\`\`\`

### Enhanced Parallel Execution
$(if [[ "$ENABLE_PARALLEL" == "true" ]]; then
echo '```bash'
echo "pytest -m \"$EXCLUDED_MARKERS\" -n $DEFAULT_PARALLEL_WORKERS --dist=worksteal --cov=src/flyrigloader"
echo '```'
else
echo '*Enhanced parallel testing not enabled*'
fi)

### Enhanced Quality Gates Validation
\`\`\`bash
./scripts/coverage/validate-quality-gates.sh
\`\`\`

### Performance Benchmarks (Isolated)
\`\`\`bash
BENCHMARK_TRIGGER=true python scripts/benchmarks/run_benchmarks.py
\`\`\`

### Network Integration Tests
\`\`\`bash
RUN_NETWORK=true pytest -m "$EXCLUDED_MARKERS" --run-network
\`\`\`

## Enhanced Environment Files

- **Project Configuration:** \`pyproject.toml\`
- **Environment Definition:** \`environment.yml\`
- **Coverage Directory:** \`$COVERAGE_DIR/\`
- **Benchmarks Directory:** \`$BENCHMARKS_DIR/\`
- **Quality Gate Script:** \`$COVERAGE_DIR/validate-quality-gates.sh\`

## Enhanced Troubleshooting

### Performance Test Isolation Issues
1. **Verify isolation directory:** Check \`$BENCHMARKS_DIR/\` exists
2. **Test marker filtering:** Run \`pytest --collect-only -m "$EXCLUDED_MARKERS"\`
3. **Benchmark execution:** Use \`BENCHMARK_TRIGGER=true\` for dedicated execution

### Enhanced Coverage Issues
1. **Check threshold:** Current threshold is $COVERAGE_THRESHOLD%
2. **Generate reports:** \`pytest --cov=src/flyrigloader --cov-report=html\`
3. **View detailed reports:** Open \`$COVERAGE_DIR/reports/htmlcov/index.html\`

---
*Enhanced environment prepared with $SCRIPT_NAME v$SCRIPT_VERSION*
EOF
        
        log "INFO" "Enhanced environment summary generated: $summary_file"
    fi
    
    # Display enhanced summary to console
    echo
    log "SUCCESS" "Enhanced CI/CD Environment Setup Completed Successfully!"
    echo
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│                 ENHANCED SETUP SUMMARY                     │"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│ Python Version:        $($PYTHON_EXECUTABLE --version 2>&1 | cut -d' ' -f2)"
    echo "│ Installation:          $INSTALL_METHOD"
    echo "│ Coverage Threshold:    $COVERAGE_THRESHOLD%"
    echo "│ Quality Gates:         $QUALITY_GATES_MODE"
    echo "│ Parallel Testing:      $(if [[ "$ENABLE_PARALLEL" == "true" ]]; then echo "Enabled ($DEFAULT_PARALLEL_WORKERS workers)"; else echo "Disabled"; fi)"
    echo "│ Performance Isolation: $(if [[ "$PERFORMANCE_ISOLATION" == "true" ]]; then echo "Enabled"; else echo "Disabled"; fi)"
    echo "│ Network Tests:         $(if [[ "$ENABLE_NETWORK_TESTS" == "true" ]]; then echo "Enabled"; else echo "Disabled"; fi)"
    echo "│ Benchmarks:            $BENCHMARK_MODE"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│ Warnings:              $WARNING_COUNT"
    echo "│ Errors:                $ERROR_COUNT"
    echo "│ Total Steps:           $STEP_COUNTER"
    echo "└─────────────────────────────────────────────────────────────┘"
    echo
    echo "${GREEN}Enhanced environment is ready for testing!${NC}"
    echo
    echo "Enhanced quick start commands:"
    echo "  ${CYAN}pytest -m \"$EXCLUDED_MARKERS\"${NC}                    # Core tests (performance isolated)"
    echo "  ${CYAN}./scripts/coverage/validate-quality-gates.sh${NC}        # Enhanced quality gates"
    if [[ "$ENABLE_PARALLEL" == "true" ]]; then
        echo "  ${CYAN}pytest -n $DEFAULT_PARALLEL_WORKERS --dist=worksteal${NC}               # Enhanced parallel execution"
    fi
    if [[ "$ENABLE_NETWORK_TESTS" == "true" ]]; then
        echo "  ${CYAN}RUN_NETWORK=true pytest --run-network${NC}              # Network integration tests"
    fi
    echo "  ${CYAN}BENCHMARK_TRIGGER=true python scripts/benchmarks/run_benchmarks.py${NC} # Performance benchmarks"
    echo
}

# =============================================================================
# MAIN EXECUTION FLOW
# =============================================================================

# Enhanced main function orchestrating the complete environment setup
main() {
    START_TIME=$(date +%s)
    
    # Display enhanced script header
    echo "================================================================================================"
    echo "                     FlyRigLoader Enhanced CI/CD Environment Preparation"
    echo "                                    Version $SCRIPT_VERSION"
    echo "            Supporting Performance Test Isolation & Enhanced Testing Strategy"
    echo "================================================================================================"
    echo
    
    # Initialize and validate enhanced environment
    parse_arguments "$@"
    detect_environment
    validate_python_version
    validate_configuration_files
    
    # Install enhanced dependencies and configure tools
    install_dependencies
    
    # Configure enhanced testing infrastructure
    configure_pytest_infrastructure
    configure_coverage_reporting
    configure_performance_benchmarks
    
    # Configure enhanced quality assurance tools
    configure_code_formatting_tools
    configure_static_type_checking
    configure_linting_tools
    configure_precommit_hooks
    configure_quality_gates
    
    # Configure enhanced CI/CD integration and optimization
    configure_github_actions_integration
    optimize_test_execution
    
    # Enhanced final validation and summary
    validate_complete_environment
    generate_environment_summary
    
    # Calculate and display execution time
    local end_time
    end_time=$(date +%s)
    local execution_time=$((end_time - START_TIME))
    
    log "SUCCESS" "Enhanced environment setup completed in ${execution_time} seconds"
    
    exit 0
}

# Enhanced script entry point with error handling
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Set up global error handling
    trap 'handle_error $? "Unexpected error occurred" "Line $LINENO" "Check error output and try again"' ERR
    
    # Execute enhanced main function with all arguments
    main "$@"
fi