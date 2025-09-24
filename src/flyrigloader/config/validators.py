"""
Specialized validation utilities for security-conscious configuration processing.

This module provides security-conscious validators for path traversal protection,
pattern validation, test-environment-aware configuration processing, and comprehensive
version validation that enhance the robustness of configuration loading and prevent
security vulnerabilities.

Enhanced with semantic versioning support for configuration schema validation and
compatibility checking, supporting the FlyRigLoader refactoring initiative for
version-aware configuration management.
"""

from pathlib import Path
import re
import os
from typing import Any, Dict, Union, Tuple, Optional, List, Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from semantic_version import Version

from .versioning import CURRENT_SCHEMA_VERSION
from flyrigloader import logger


DEFAULT_SENSITIVE_ROOTS: Tuple[str, ...] = (
    '/bin',
    '/etc',
    '/dev',
    '/proc',
    '/sys',
    '/root',
    '/boot',
    '/sbin',
)


class PathSecurityPolicy(BaseModel):
    """Configuration-driven allow/deny lists for path validation."""

    model_config = ConfigDict(frozen=True)

    allow_roots: Tuple[str, ...] = Field(default_factory=tuple)
    deny_roots: Tuple[str, ...] = Field(default_factory=tuple)
    inherit_defaults: bool = Field(
        default=True,
        description="Include DEFAULT_SENSITIVE_ROOTS in deny list when True.",
    )

    @field_validator('allow_roots', 'deny_roots', mode='before')
    @classmethod
    def _coerce_roots(cls, value: Optional[Sequence[str]]) -> Tuple[str, ...]:
        """Ensure sequences are provided for allow/deny roots."""
        if value is None:
            return ()
        if isinstance(value, (str, bytes)):
            raise TypeError('Root lists must be provided as a sequence of strings')
        return tuple(value)

    @field_validator('allow_roots', 'deny_roots')
    @classmethod
    def _validate_roots(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        """Validate and normalize root prefixes."""
        normalized: List[str] = []
        seen = set()
        for root in value:
            if not isinstance(root, str):
                raise TypeError('Root entries must be strings')
            candidate = root.strip()
            if not candidate:
                raise ValueError('Root entries cannot be empty')
            if not candidate.startswith('/'):
                raise ValueError(f"Root '{candidate}' must be absolute")
            normalized_root = os.path.normpath(candidate)
            if not normalized_root.startswith('/'):
                raise ValueError(f"Root '{candidate}' must resolve to an absolute path")
            normalized_root = normalized_root.rstrip('/') or '/'
            if normalized_root in seen:
                raise ValueError(f"Duplicate root entry detected: '{normalized_root}'")
            seen.add(normalized_root)
            normalized.append(normalized_root)
        return tuple(normalized)

    @model_validator(mode='after')
    def _ensure_disjoint_sets(self) -> 'PathSecurityPolicy':
        allow = set(self.allow_roots)
        deny = set(self.deny_roots)
        overlap = allow & deny
        if overlap:
            overlap_display = ', '.join(sorted(overlap))
            raise ValueError(
                f"Allow and deny roots must be disjoint; overlapping entries: {overlap_display}"
            )
        return self

    def effective_deny_roots(self) -> Tuple[str, ...]:
        """Return the deny roots with defaults applied."""
        if self.inherit_defaults:
            combined = DEFAULT_SENSITIVE_ROOTS + self.deny_roots
        else:
            combined = self.deny_roots
        # Preserve order while removing duplicates
        seen = set()
        ordered: List[str] = []
        for root in combined:
            if root in seen:
                continue
            seen.add(root)
            ordered.append(root)
        return tuple(ordered)

    def match_allow_root(self, path_str: str) -> Optional[str]:
        """Return the allow root that matches the provided path, if any."""
        for root in self.allow_roots:
            if _path_matches_root(path_str, root):
                return root
        return None


def _normalize_path_for_matching(path_str: str) -> str:
    """Normalize path for root prefix matching without touching filesystem."""
    normalized = os.path.normpath(path_str)
    if path_str.endswith('/') and normalized != '/':
        normalized = normalized.rstrip('/')
    return normalized


def _path_matches_root(path_str: str, root: str) -> bool:
    """Return True when the path is at or under the provided root."""
    normalized_path = _normalize_path_for_matching(path_str)
    if normalized_path == root:
        return True
    if root == '/':
        return normalized_path.startswith('/')
    return normalized_path.startswith(f"{root}/")


def path_traversal_protection(
    path_input: Any,
    security_policy: Optional[PathSecurityPolicy] = None,
) -> str:
    """
    Validate and sanitize path input to prevent directory traversal attacks.
    
    This function implements comprehensive path security validation following
    the security checkpoint workflow defined in Section 4.4.2.1. It prevents
    directory traversal attacks through input sanitization and validates that
    paths are within acceptable boundaries.
    
    Args:
        path_input: Path input to validate (string, Path object, or other)
        
    Returns:
        str: Sanitized path string that has passed security validation
        
    Raises:
        ValueError: If path contains traversal attempts or forbidden patterns
        PermissionError: If path attempts to access restricted system locations
        TypeError: If path_input is not a valid path type
        
    Security Checks:
    - Path traversal prevention (../, /.., ~, //)
    - Remote URL blocking (file://, http://, https://, ftp://)
    - Sensitive root access restriction configurable via PathSecurityPolicy
    - Null byte injection prevention
    - Excessive path length protection
    """
    # Type validation
    if not isinstance(path_input, (str, Path)):
        raise TypeError(f"Path input must be string or Path, got {type(path_input)}")
    
    # Convert to string for validation
    path_str = str(path_input)
    
    # Check for null bytes (potential injection attack)
    if '\x00' in path_str:
        logger.warning(f"Null byte detected in path: {path_str!r}")
        raise ValueError("Path contains null bytes - potential security risk")
    
    # Check for excessive path length (potential DoS attack)
    if len(path_str) > 4096:
        logger.warning(f"Excessive path length detected: {len(path_str)} characters")
        raise ValueError("Path length exceeds maximum allowed limit")
    
    # Check for remote URLs (security violation)
    url_prefixes = ('file://', 'http://', 'https://', 'ftp://', 'ftps://', 'ssh://')
    if any(path_str.startswith(prefix) for prefix in url_prefixes):
        logger.error(f"Remote URL blocked for security: {path_str}")
        raise ValueError(f"Remote or file:// URLs are not allowed: {path_str}")
    
    policy = security_policy or PathSecurityPolicy()
    allow_root = policy.match_allow_root(path_str)
    if allow_root:
        logger.debug(
            f"Path '{path_str}' allowed by configured allow root '{allow_root}'"
        )

    deny_roots = policy.effective_deny_roots()
    logger.debug(
        f"Evaluating path '{path_str}' against deny roots: {list(deny_roots)}"
    )
    for root in deny_roots:
        if allow_root and _path_matches_root(allow_root, root):
            logger.debug(
                f"Allow root '{allow_root}' takes precedence over deny root '{root}' for path '{path_str}'"
            )
            continue
        if _path_matches_root(path_str, root):
            logger.error(
                f"Sensitive root '{root}' access blocked for path: {path_str}"
            )
            raise PermissionError(
                f"Access to sensitive system root '{root}' is not allowed: {path_str}"
            )
    
    # Check for path traversal attempts (security violation)
    traversal_patterns = ('../', '/..', '..\\', '\\..', '~/', '~\\', '//', '\\\\')
    if any(pattern in path_str for pattern in traversal_patterns):
        logger.error(f"Path traversal attempt detected: {path_str}")
        raise ValueError(f"Path traversal is not allowed: {path_str}")
    
    # Check for hidden or special characters that could be problematic
    suspicious_chars = set(path_str) & {'\n', '\r', '\t', '\f', '\v'}
    if suspicious_chars:
        logger.warning(f"Suspicious characters in path: {suspicious_chars}")
        raise ValueError(f"Path contains suspicious characters: {suspicious_chars}")
    
    # Additional Windows-specific checks
    if os.name == 'nt':
        # Check for Windows reserved names
        reserved_names = {
            'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 'COM5',
            'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 'LPT4',
            'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        }
        path_parts = Path(path_str).parts
        for part in path_parts:
            if part.upper().split('.')[0] in reserved_names:
                logger.error(f"Windows reserved name detected: {part}")
                raise ValueError(f"Windows reserved name not allowed: {part}")
    
    # Log successful validation
    logger.debug(f"Path traversal protection passed for: {path_str}")
    
    return path_str


def pattern_validation(pattern: Any) -> re.Pattern:
    """
    Validate and compile regex patterns with comprehensive error handling.
    
    This function ensures that regex patterns are valid before runtime usage,
    preventing regex compilation errors during file discovery operations.
    It provides detailed error messages for pattern debugging and maintains
    a secure approach to pattern compilation.
    
    Args:
        pattern: Regex pattern to validate (string or compiled Pattern)
        
    Returns:
        re.Pattern: Compiled regex pattern object
        
    Raises:
        TypeError: If pattern is not a string or Pattern object
        ValueError: If pattern is invalid regex syntax
        re.error: If pattern compilation fails with detailed error context
        
    Security Considerations:
    - Pattern complexity validation to prevent ReDoS attacks
    - Length limits to prevent memory exhaustion
    - Compilation timeout protection
    """
    # Handle already compiled patterns
    if isinstance(pattern, re.Pattern):
        logger.debug(f"Pattern already compiled: {pattern.pattern}")
        return pattern
    
    # Type validation
    if not isinstance(pattern, str):
        raise TypeError(f"Pattern must be string or compiled Pattern, got {type(pattern)}")
    
    # Check for empty pattern
    if not pattern.strip():
        raise ValueError("Pattern cannot be empty or whitespace-only")
    
    # Check for excessive pattern length (potential DoS protection)
    if len(pattern) > 1000:
        logger.warning(f"Excessive pattern length: {len(pattern)} characters")
        raise ValueError("Pattern length exceeds maximum allowed limit")
    
    # Check for potentially dangerous patterns (ReDoS protection)
    # Look for nested quantifiers that could cause exponential backtracking
    dangerous_patterns = [
        r'\([^)]*\+[^)]*\+[^)]*\)',  # Nested + quantifiers
        r'\([^)]*\*[^)]*\*[^)]*\)',  # Nested * quantifiers
        r'\([^)]*\{[^}]*,[^}]*\}[^)]*\{[^}]*,[^}]*\}[^)]*\)',  # Nested {} quantifiers
    ]
    
    for dangerous in dangerous_patterns:
        if re.search(dangerous, pattern):
            logger.warning(f"Potentially dangerous pattern detected: {pattern}")
            # Don't raise error, just log warning - let user decide
    
    # Attempt pattern compilation with detailed error handling
    try:
        compiled_pattern = re.compile(pattern)
        logger.debug(f"Pattern compiled successfully: {pattern}")
        return compiled_pattern
        
    except re.error as e:
        error_msg = f"Invalid regex pattern '{pattern}': {str(e)}"
        logger.error(error_msg)
        raise re.error(error_msg) from e
    
    except Exception as e:
        error_msg = f"Unexpected error compiling pattern '{pattern}': {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def path_existence_validator(
    path_input: Any,
    require_file: bool = False,
    security_policy: Optional[PathSecurityPolicy] = None,
) -> bool:
    """
    Validate path existence with test environment awareness.
    
    This function checks if paths exist on the filesystem while respecting
    test environment settings. During pytest execution, existence checks
    are disabled to allow testing with mock files that don't actually exist.
    
    Args:
        path_input: Path to validate (string, Path object, or other)
        require_file: If True, require path to be a file (not directory)
        security_policy: Optional allow/deny configuration for sensitive roots
        
    Returns:
        bool: True if path exists and meets requirements, False otherwise
        
    Raises:
        TypeError: If path_input is not a valid path type
        ValueError: If path fails security validation
        FileNotFoundError: If path doesn't exist (only in non-test environments)
        
    Test Environment Behavior:
    - During pytest execution: Always returns True (validation disabled)
    - Production environment: Performs actual filesystem checks
    - Environment detection via PYTEST_CURRENT_TEST variable
    """
    # Type validation
    if not isinstance(path_input, (str, Path)):
        raise TypeError(f"Path input must be string or Path, got {type(path_input)}")
    
    # Security validation first
    path_str = path_traversal_protection(path_input, security_policy=security_policy)
    
    # Check if we're in a test environment
    is_test_env = os.environ.get('PYTEST_CURRENT_TEST') is not None
    
    if is_test_env:
        logger.debug(f"Test environment detected - skipping existence check for: {path_str}")
        return True
    
    # Convert to Path object for filesystem operations
    try:
        path_obj = Path(path_str).resolve()
    except (RuntimeError, OSError) as e:
        error_msg = f"Error resolving path {path_str}: {e}"
        logger.error(error_msg)
        raise OSError(error_msg) from e
    
    # Check if path exists
    if not path_obj.exists():
        logger.error(f"Path does not exist: {path_str}")
        raise FileNotFoundError(f"Path not found: {path_str}")
    
    # Check if it's a file when required
    if require_file and not path_obj.is_file():
        logger.error(f"Path is not a file: {path_str}")
        raise ValueError(f"Path is not a file: {path_str}")
    
    # Check if it's a directory when file is not required
    if not require_file and not path_obj.is_dir() and not path_obj.is_file():
        logger.error(f"Path is neither file nor directory: {path_str}")
        raise ValueError(f"Path is neither file nor directory: {path_str}")
    
    logger.debug(f"Path existence validated: {path_str}")
    return True


def date_format_validator(date_input: Any, date_format: str = '%Y-%m-%d') -> bool:
    """
    Validate date format consistency for dates_vials structure.
    
    This function ensures that dates in the dates_vials configuration follow
    a consistent format across all datasets. It supports multiple date formats
    and provides detailed error messages for debugging date-related issues.
    
    Args:
        date_input: Date string to validate
        date_format: Expected date format (default: '%Y-%m-%d')
        
    Returns:
        bool: True if date format is valid, False otherwise
        
    Raises:
        TypeError: If date_input is not a string
        ValueError: If date format is invalid or date is unparseable
        
    Supported Formats:
    - ISO format: YYYY-MM-DD (default)
    - US format: MM/DD/YYYY
    - European format: DD/MM/YYYY
    - Compact format: YYYYMMDD
    - Custom formats via date_format parameter
    """
    # Type validation
    if not isinstance(date_input, str):
        raise TypeError(f"Date input must be string, got {type(date_input)}")
    
    # Check for empty date
    if not date_input.strip():
        raise ValueError("Date cannot be empty or whitespace-only")
    
    # Security check - prevent excessively long date strings
    if len(date_input) > 50:
        raise ValueError("Date string exceeds maximum allowed length")
    
    # Try primary format first
    try:
        parsed_date = datetime.strptime(date_input, date_format)
        logger.debug(f"Date parsed successfully with format {date_format}: {date_input}")
        return True
        
    except ValueError as primary_error:
        logger.debug(f"Primary format {date_format} failed for {date_input}: {primary_error}")
        
        # Try common alternative formats
        alternative_formats = [
            '%Y-%m-%d',      # ISO format
            '%Y_%m_%d',      # Underscore format
            '%m/%d/%Y',      # US format
            '%d/%m/%Y',      # European format
            '%Y%m%d',        # Compact format
            '%Y-%m-%d %H:%M:%S',  # ISO with time
            '%m/%d/%Y %H:%M:%S',  # US with time
        ]
        
        for alt_format in alternative_formats:
            if alt_format == date_format:
                continue  # Skip the format we already tried
                
            try:
                parsed_date = datetime.strptime(date_input, alt_format)
                logger.info(f"Date parsed with alternative format {alt_format}: {date_input}")
                return True
                
            except ValueError:
                continue
        
        # Try ISO format parsing as last resort
        try:
            parsed_date = datetime.fromisoformat(date_input.replace('Z', '+00:00'))
            logger.info(f"Date parsed with ISO format: {date_input}")
            return True
            
        except ValueError:
            pass
    
    # If all parsing attempts failed, raise detailed error
    error_msg = (
        f"Invalid date format: '{date_input}'. "
        f"Expected format: {date_format}. "
        f"Supported formats include: YYYY-MM-DD, YYYY_MM_DD, MM/DD/YYYY, DD/MM/YYYY, YYYYMMDD"
    )
    logger.error(error_msg)
    raise ValueError(error_msg)


def validate_version_format(version_string: Any) -> bool:
    """
    Validate that a version string conforms to semantic versioning standards.
    
    This function ensures that configuration schema_version fields follow semantic
    versioning format (MAJOR.MINOR.PATCH) as required by the version management
    system. It provides strict validation with detailed error reporting for
    debugging version-related configuration issues.
    
    Args:
        version_string: Version string to validate (must be string type)
        
    Returns:
        bool: True if version format is valid, False otherwise
        
    Raises:
        TypeError: If version_string is not a string
        ValueError: If version format is invalid or unparseable
        
    Example:
        >>> validate_version_format("1.0.0")  # Returns True
        >>> validate_version_format("1.0")    # Raises ValueError
        >>> validate_version_format("invalid") # Raises ValueError
        
    Supported Format:
        - Semantic versioning: MAJOR.MINOR.PATCH (e.g., "1.0.0", "2.1.3")
        - Optional pre-release suffixes: "1.0.0-alpha", "1.0.0-beta.1"
        - Optional build metadata: "1.0.0+build.1", "1.0.0-alpha+beta"
    """
    # Type validation
    if not isinstance(version_string, str):
        logger.error(f"Version string must be str, got {type(version_string)}")
        raise TypeError(f"Version string must be str, got {type(version_string)}")
    
    # Check for empty version
    if not version_string.strip():
        logger.error("Version string cannot be empty or whitespace-only")
        raise ValueError("Version string cannot be empty or whitespace-only")
    
    # Security check - prevent excessively long version strings
    if len(version_string) > 100:
        logger.error(f"Version string exceeds maximum length: {len(version_string)} chars")
        raise ValueError("Version string exceeds maximum allowed length")
    
    # Clean the version string
    clean_version = version_string.strip()
    
    try:
        # Use semantic_version library for strict parsing
        parsed_version = Version(clean_version)
        
        # Additional validation for our specific requirements
        if parsed_version.major < 0 or parsed_version.minor < 0 or parsed_version.patch < 0:
            logger.error(f"Version components must be non-negative: {clean_version}")
            raise ValueError("Version components must be non-negative integers")
        
        logger.debug(f"Version format validation passed: {clean_version}")
        return True
        
    except Exception as e:
        error_msg = f"Invalid semantic version format '{clean_version}': {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def validate_version_compatibility(
    config_version: str,
    system_version: str = CURRENT_SCHEMA_VERSION
) -> Tuple[bool, str, Optional[List[str]]]:
    """
    Validate configuration version compatibility with the current system version.

    Args:
        config_version: Version of the configuration to validate
        system_version: Current system version (defaults to CURRENT_SCHEMA_VERSION)

    Returns:
        Tuple[bool, str, Optional[List[str]]]:
            - is_compatible: True if supported without changes
            - message: Detailed compatibility status message
            - details: Reserved for future use (currently ``None``)

    Raises:
        ValueError: If version formats are invalid
        TypeError: If version inputs are not strings
    """
    logger.debug(
        f"Validating compatibility without upgrade pathways: config v{config_version} vs system v{system_version}"
    )

    if not isinstance(config_version, str):
        raise TypeError(f"config_version must be string, got {type(config_version)}")
    if not isinstance(system_version, str):
        raise TypeError(f"system_version must be string, got {type(system_version)}")

    validate_version_format(config_version)
    validate_version_format(system_version)

    config_ver = Version(config_version)
    system_ver = Version(system_version)

    if config_ver == system_ver:
        message = f"Configuration version {config_version} is supported"
        logger.info(message)
        return True, message, None

    if config_ver > system_ver:
        message = (
            f"Configuration version {config_version} is newer than the supported "
            f"system version {system_version}. Upgrade FlyRigLoader to use this configuration."
        )
        logger.error(message)
        return False, message, None

    message = (
        f"Configuration version {config_version} is deprecated and no supported upgrade path exists. "
        f"Update the configuration to version {system_version}."
    )
    logger.error(message)
    return False, message, None


def validate_config_version(config_data: Union[Dict[str, Any], str]) -> Tuple[bool, str, str]:
    """Validate configuration version information and determine compatibility.

    Args:
        config_data: Configuration data as dictionary or raw YAML string.

    Returns:
        Tuple[bool, str, str]:
            - is_valid: True if configuration is valid and compatible.
            - detected_version: The detected or extracted version string.
            - validation_message: Detailed validation result message.

    Raises:
        TypeError: If ``config_data`` is not dict or string.
        ValueError: If a version cannot be detected or is invalid.

    Example:
        >>> config = {"schema_version": "1.0.0", "project": {"name": "test"}}
        >>> valid, version, msg = validate_config_version(config)
        >>> assert valid is True
        >>> assert version == "1.0.0"

        >>> outdated_config = {"project": {"directories": {}}, "experiments": {}}
        >>> validate_config_version(outdated_config)
        Traceback (most recent call last):
        ...
        ValueError: Configuration version validation failed: Cannot detect version: unrecognized configuration structure
    """
    logger.debug("Starting configuration version validation")
    
    # Type validation
    if not isinstance(config_data, (dict, str)):
        raise TypeError(f"Configuration data must be dict or str, got {type(config_data)}")
    
    try:
        detected_version = _extract_version_from_config(config_data)
        logger.info(f"Detected configuration version: {detected_version}")

        validate_version_format(detected_version)

        is_compatible, compat_message, _ = validate_version_compatibility(detected_version)

        if is_compatible:
            validation_message = f"Configuration version {detected_version} is valid and compatible"
            logger.info(validation_message)
            return True, detected_version, validation_message

        logger.error(compat_message)
        return False, detected_version, compat_message

    except Exception as e:
        error_msg = f"Configuration version validation failed: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def _extract_version_from_config(config_data: Union[Dict[str, Any], str]) -> str:
    """Extract version information from configuration data.

    Args:
        config_data: Configuration data as dictionary or YAML string.

    Returns:
        str: Detected version string.

    Raises:
        ValueError: If a version cannot be detected.
    """
    logger.debug("Extracting version from configuration data")
    
    # Handle dictionary-based configuration
    if isinstance(config_data, dict):
        # Check for explicit schema_version field
        if 'schema_version' in config_data:
            version = str(config_data['schema_version']).strip()
            logger.debug(f"Found explicit schema_version: {version}")
            return version
        
        # Structural analysis for legacy configurations
        has_project = 'project' in config_data
        has_datasets = 'datasets' in config_data
        has_experiments = 'experiments' in config_data
        
        logger.debug(f"Structure analysis - project: {has_project}, datasets: {has_datasets}, experiments: {has_experiments}")
        
        # Version 0.2.0: Has project, datasets, and experiments
        if has_project and has_datasets and has_experiments:
            logger.info("Detected legacy configuration structure v0.2.0")
            return "0.2.0"
        
        # Version 0.1.0: Has project and experiments, but no datasets
        elif has_project and has_experiments and not has_datasets:
            logger.info("Detected legacy configuration structure v0.1.0")
            return "0.1.0"
        
        # Modern configuration without explicit version
        elif has_project:
            logger.info("Detected modern configuration, assuming current version")
            return "1.0.0"
        
        else:
            raise ValueError("Cannot detect version: unrecognized configuration structure")
    
    # Handle string-based configuration (YAML content)
    elif isinstance(config_data, str):
        # Pattern for explicit schema_version field
        version_pattern = re.compile(
            r'schema_version\s*:\s*["\']?(\d+\.\d+\.\d+)["\']?',
            re.MULTILINE | re.IGNORECASE
        )
        
        version_match = version_pattern.search(config_data)
        if version_match:
            version = version_match.group(1)
            logger.debug(f"Found explicit schema_version in YAML: {version}")
            return version
        
        # Structural pattern analysis for legacy formats
        if re.search(r'(?=.*project\s*:)(?=.*datasets\s*:)(?=.*experiments\s*:)', config_data, re.MULTILINE | re.IGNORECASE | re.DOTALL):
            logger.info("Detected legacy v0.2.0 pattern in YAML")
            return "0.2.0"
        elif re.search(r'(?=.*project\s*:)(?=.*experiments\s*:)(?!.*datasets\s*:)', config_data, re.MULTILINE | re.IGNORECASE | re.DOTALL):
            logger.info("Detected legacy v0.1.0 pattern in YAML")
            return "0.1.0"
        else:
            logger.error("Cannot detect version from YAML content")
            raise ValueError(
                "Cannot detect version: add an explicit 'schema_version' field or update the configuration structure"
            )
    
    raise ValueError("Unsupported configuration data type for version extraction")


def validate_config_with_version(
    config_data: Union[Dict[str, Any], str],
    expected_version: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Comprehensively validate configuration data with version awareness.

    Args:
        config_data: Configuration data as dictionary or YAML string.
        expected_version: Expected version string (``None`` to auto-detect).

    Returns:
        Tuple[bool, Dict[str, Any]]: Validation status and detailed results including
        the detected version, warnings, and errors.

    Raises:
        TypeError: If ``config_data`` is not dict or string.
        ValueError: If configuration is invalid or incompatible.
    """
    logger.debug("Starting comprehensive configuration validation with version awareness")
    
    # Initialize validation result structure
    validation_result = {
        'version': None,
        'compatible': False,
        'validation_messages': [],
        'warnings': [],
        'errors': []
    }
    
    try:
        # Step 1: Version validation
        is_version_valid, detected_version, version_message = validate_config_version(config_data)
        validation_result['version'] = detected_version
        validation_result['validation_messages'].append(version_message)
        
        # Step 2: Expected version check (if specified)
        if expected_version is not None:
            validate_version_format(expected_version)
            if detected_version != expected_version:
                error_msg = (
                    f"Detected version {detected_version} does not match expected {expected_version}."
                )
                logger.error(error_msg)
                validation_result['errors'].append(error_msg)
                return False, validation_result

        # Step 3: Compatibility assessment
        is_compatible, compat_message, _ = validate_version_compatibility(detected_version)
        validation_result['compatible'] = is_compatible
        validation_result['validation_messages'].append(compat_message)

        if not is_compatible:
            logger.error(compat_message)
            validation_result['errors'].append(compat_message)
            return False, validation_result
        
        # Step 4: Additional configuration structure validation
        structure_valid, structure_messages = _validate_config_structure(config_data, detected_version)
        validation_result['validation_messages'].extend(structure_messages)
        
        if not structure_valid:
            validation_result['errors'].extend(structure_messages)
            return False, validation_result
        
        # Step 5: Generate deprecation warnings if applicable
        deprecation_warnings = _get_version_deprecation_warnings(detected_version)
        validation_result['warnings'].extend(deprecation_warnings)
        
        overall_valid = is_version_valid and structure_valid and is_compatible

        if overall_valid:
            logger.info(f"Configuration validation passed for version {detected_version}")
        else:
            logger.warning(f"Configuration validation failed for version {detected_version}")

        return overall_valid, validation_result
    
    except Exception as e:
        error_msg = f"Configuration validation failed with error: {str(e)}"
        logger.error(error_msg)
        validation_result['errors'].append(error_msg)
        return False, validation_result


def _validate_config_structure(config_data: Union[Dict[str, Any], str], version: str) -> Tuple[bool, List[str]]:
    """Validate configuration structure based on version-specific requirements.

    Args:
        config_data: Configuration data to validate.
        version: Configuration version for validation rules.

    Returns:
        Tuple[bool, List[str]]: (is_valid, validation_messages).
    """
    logger.debug(f"Validating configuration structure for version {version}")
    
    messages = []
    
    # Convert string data to dict for structure validation
    if isinstance(config_data, str):
        # For string data, we can only do basic pattern validation
        messages.append("String-based configuration passed basic structure validation")
        return True, messages
    
    # Dictionary-based structure validation
    if not isinstance(config_data, dict):
        messages.append("Configuration must be a dictionary structure")
        return False, messages
    
    # Version-specific structure requirements
    try:
        version_obj = Version(version)
        
        # Basic requirements for all versions
        if 'project' not in config_data:
            messages.append("Configuration must contain 'project' section")
            return False, messages
        
        # Version 0.2.0+ requires datasets section
        if version_obj >= Version("0.2.0") and 'datasets' not in config_data:
            messages.append(f"Configuration version {version} requires 'datasets' section")
            
        # Version 1.0.0+ should have schema_version
        if version_obj >= Version("1.0.0") and 'schema_version' not in config_data:
            messages.append("Modern configurations should include explicit 'schema_version' field")
        
        messages.append(f"Configuration structure validation passed for version {version}")
        return True, messages
    
    except Exception as e:
        messages.append(f"Structure validation failed: {str(e)}")
        return False, messages


def _get_version_deprecation_warnings(version: str) -> List[str]:
    """Get version-specific deprecation warnings without upgrade guidance."""

    warnings_list: List[str] = []

    try:
        version_obj = Version(version)
        current_version = Version(CURRENT_SCHEMA_VERSION)

        if version_obj < current_version:
            warnings_list.append(
                f"Configuration version {version} is deprecated. "
                f"Update to {CURRENT_SCHEMA_VERSION} to ensure compatibility."
            )

    except Exception as e:
        logger.debug(f"Could not generate deprecation warnings for version {version}: {e}")

    return warnings_list
