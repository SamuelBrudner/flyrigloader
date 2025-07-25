"""
Version-specific validation logic for FlyRigLoader configuration schemas across different versions.

This module provides comprehensive validation functions for legacy configuration formats, schema
compatibility checking, and version-aware validation with automatic fallback handling. It ensures
configuration integrity during migration while maintaining security and path validation consistency
with existing validator infrastructure.

The module implements the version management system defined in Section 5.3.6, providing validation
functions for v0.1.0, v0.2.0, and v1.0.0 configurations with automatic migration support and
comprehensive error reporting for research workflow continuity.
"""

import logging
import re
import warnings
from typing import Any, Dict, List, Optional, Union, Tuple, Callable

from semantic_version import Version
from pydantic import ValidationError

# Internal imports for version detection and validation consistency
from ..migration.versions import detect_config_version, ConfigVersion, CURRENT_VERSION



# Set up logger for version validation operations
logger = logging.getLogger(__name__)


def validate_v0_1_0_config(config_data: Dict[str, Any]) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    """
    Validate legacy v0.1.0 configuration format with basic structure checking.
    
    Version 0.1.0 configurations contain only project and experiments sections without
    datasets support or comprehensive validation. This function ensures backward compatibility
    while identifying potential upgrade opportunities.
    
    Args:
        config_data: Configuration dictionary to validate in v0.1.0 format
        
    Returns:
        Tuple[bool, List[str], Optional[Dict[str, Any]]]: 
            (is_valid, validation_errors, sanitized_config)
            
    Raises:
        TypeError: If config_data is not a dictionary
        
    Example:
        >>> config = {"project": {"directories": {}}, "experiments": {"exp1": {}}}
        >>> valid, errors, sanitized = validate_v0_1_0_config(config)
        >>> assert valid == True
    """
    logger.debug("Validating configuration against v0.1.0 schema")
    
    # Input type validation
    if not isinstance(config_data, dict):
        logger.error(f"Configuration must be dictionary, got {type(config_data)}")
        raise TypeError(f"Configuration data must be dictionary, got {type(config_data)}")
    
    validation_errors = []
    sanitized_config = {}
    
    try:
        # Emit deprecation warning for v0.1.0 usage
        warnings.warn(
            "Configuration version 0.1.0 is deprecated. Missing datasets section limits "
            "experimental flexibility. Consider upgrading to v1.0.0 for full feature support.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Validate required top-level structure
        required_sections = ['project', 'experiments']
        for section in required_sections:
            if section not in config_data:
                validation_errors.append(f"Missing required section: '{section}'")
                logger.warning(f"v0.1.0 validation failed: missing section '{section}'")
                continue
            
            if not isinstance(config_data[section], dict):
                validation_errors.append(f"Section '{section}' must be a dictionary")
                logger.warning(f"v0.1.0 validation failed: '{section}' is not dictionary")
                continue
            
            sanitized_config[section] = config_data[section].copy()
        
        # Validate project section structure (basic v0.1.0 requirements)
        if 'project' in config_data:
            project_errors = _validate_v0_1_0_project_section(config_data['project'])
            validation_errors.extend(project_errors)
        
        # Validate experiments section structure
        if 'experiments' in config_data:
            experiments_errors = _validate_v0_1_0_experiments_section(config_data['experiments'])
            validation_errors.extend(experiments_errors)
        
        # Check for unsupported v0.1.0 features
        if 'datasets' in config_data:
            validation_errors.append(
                "Datasets section not supported in v0.1.0. Consider upgrading to v0.2.0 or v1.0.0"
            )
            logger.info("Configuration contains datasets section - suggests v0.2.0+ format")
        
        # Copy any additional sections for forward compatibility
        for key, value in config_data.items():
            if key not in ['project', 'experiments', 'datasets']:
                sanitized_config[key] = value
                logger.debug(f"Preserved additional section: {key}")
        
        is_valid = len(validation_errors) == 0
        
        if is_valid:
            logger.info("v0.1.0 configuration validation passed successfully")
        else:
            logger.warning(f"v0.1.0 configuration validation failed with {len(validation_errors)} errors")
        
        return is_valid, validation_errors, sanitized_config if is_valid else None
        
    except Exception as e:
        logger.error(f"Unexpected error during v0.1.0 validation: {e}")
        validation_errors.append(f"Validation error: {str(e)}")
        return False, validation_errors, None


def validate_v0_2_0_config(config_data: Dict[str, Any]) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    """
    Validate legacy v0.2.0 configuration format with enhanced structure checking.
    
    Version 0.2.0 configurations include project, datasets, and experiments sections with
    basic validation support. This function provides comprehensive validation while preparing
    for potential migration to v1.0.0 Pydantic-based validation.
    
    Args:
        config_data: Configuration dictionary to validate in v0.2.0 format
        
    Returns:
        Tuple[bool, List[str], Optional[Dict[str, Any]]]: 
            (is_valid, validation_errors, sanitized_config)
            
    Raises:
        TypeError: If config_data is not a dictionary
        
    Example:
        >>> config = {
        ...     "project": {"directories": {}},
        ...     "datasets": {"ds1": {"rig": "rig1"}},
        ...     "experiments": {"exp1": {"datasets": ["ds1"]}}
        ... }
        >>> valid, errors, sanitized = validate_v0_2_0_config(config)
        >>> assert valid == True
    """
    logger.debug("Validating configuration against v0.2.0 schema")
    
    # Input type validation
    if not isinstance(config_data, dict):
        logger.error(f"Configuration must be dictionary, got {type(config_data)}")
        raise TypeError(f"Configuration data must be dictionary, got {type(config_data)}")
    
    validation_errors = []
    sanitized_config = {}
    
    try:
        # Emit deprecation warning for v0.2.0 usage
        warnings.warn(
            "Configuration version 0.2.0 is deprecated. Limited error handling compared to "
            "modern validation. Upgrade to v1.0.0 recommended for full feature support.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Validate required top-level structure for v0.2.0
        required_sections = ['project', 'datasets', 'experiments']
        for section in required_sections:
            if section not in config_data:
                validation_errors.append(f"Missing required section: '{section}'")
                logger.warning(f"v0.2.0 validation failed: missing section '{section}'")
                continue
            
            if not isinstance(config_data[section], dict):
                validation_errors.append(f"Section '{section}' must be a dictionary")
                logger.warning(f"v0.2.0 validation failed: '{section}' is not dictionary")
                continue
            
            sanitized_config[section] = config_data[section].copy()
        
        # Validate project section (enhanced v0.2.0 requirements)
        if 'project' in config_data:
            project_errors = _validate_v0_2_0_project_section(config_data['project'])
            validation_errors.extend(project_errors)
        
        # Validate datasets section (new in v0.2.0)
        if 'datasets' in config_data:
            datasets_errors = _validate_v0_2_0_datasets_section(config_data['datasets'])
            validation_errors.extend(datasets_errors)
        
        # Validate experiments section with dataset references
        if 'experiments' in config_data:
            experiments_errors = _validate_v0_2_0_experiments_section(
                config_data['experiments'], 
                list(config_data.get('datasets', {}).keys())
            )
            validation_errors.extend(experiments_errors)
        
        # Copy any additional sections for forward compatibility
        for key, value in config_data.items():
            if key not in ['project', 'datasets', 'experiments']:
                sanitized_config[key] = value
                logger.debug(f"Preserved additional section: {key}")
        
        is_valid = len(validation_errors) == 0
        
        if is_valid:
            logger.info("v0.2.0 configuration validation passed successfully")
        else:
            logger.warning(f"v0.2.0 configuration validation failed with {len(validation_errors)} errors")
        
        return is_valid, validation_errors, sanitized_config if is_valid else None
        
    except Exception as e:
        logger.error(f"Unexpected error during v0.2.0 validation: {e}")
        validation_errors.append(f"Validation error: {str(e)}")
        return False, validation_errors, None


def validate_schema_compatibility(
    config_data: Dict[str, Any], 
    target_version: str = CURRENT_VERSION
) -> Tuple[bool, str, List[str]]:
    """
    Validate schema compatibility between configuration data and target version.
    
    This function analyzes configuration structure and content to determine compatibility
    with a target schema version, identifying required migrations and potential conflicts.
    It implements the compatibility matrix validation defined in Section 5.3.6.
    
    Args:
        config_data: Configuration dictionary to validate
        target_version: Target schema version for compatibility check
        
    Returns:
        Tuple[bool, str, List[str]]: (is_compatible, detected_version, compatibility_issues)
        
    Raises:
        ValueError: If target_version is not a valid semantic version
        TypeError: If config_data is not a dictionary
        
    Example:
        >>> config = {"schema_version": "0.1.0", "project": {}, "experiments": {}}
        >>> compatible, version, issues = validate_schema_compatibility(config, "1.0.0")
        >>> assert compatible == False  # Migration needed
    """
    logger.debug(f"Validating schema compatibility with target version {target_version}")
    
    # Input validation
    if not isinstance(config_data, dict):
        logger.error(f"Configuration must be dictionary, got {type(config_data)}")
        raise TypeError(f"Configuration data must be dictionary, got {type(config_data)}")
    
    try:
        # Validate target version format
        target_sem_version = Version(target_version)
        logger.debug(f"Target version parsed successfully: {target_sem_version}")
    except Exception as e:
        logger.error(f"Invalid target version format '{target_version}': {e}")
        raise ValueError(f"Invalid semantic version format: {target_version}") from e
    
    compatibility_issues = []
    
    try:
        # Detect current configuration version
        detected_version = detect_config_version(config_data)
        detected_version_str = detected_version.value
        
        logger.info(f"Detected configuration version: {detected_version_str}")
        
        # Same version - fully compatible
        if detected_version_str == target_version:
            logger.info("Configuration version matches target version - fully compatible")
            return True, detected_version_str, []
        
        # Version comparison using semantic versioning
        detected_sem_version = Version(detected_version_str)
        
        # Configuration is newer than target - potential compatibility issues
        if detected_sem_version > target_sem_version:
            compatibility_issues.append(
                f"Configuration version {detected_version_str} is newer than target {target_version}"
            )
            compatibility_issues.append(
                "Newer configuration may contain unsupported features"
            )
            logger.warning(f"Configuration version {detected_version_str} > target {target_version}")
            return False, detected_version_str, compatibility_issues
        
        # Configuration is older - check migration availability
        from ..migration.versions import is_migration_available, get_deprecation_warnings
        
        if is_migration_available(detected_version_str, target_version):
            compatibility_issues.append(
                f"Configuration version {detected_version_str} can be migrated to {target_version}"
            )
            
            # Add deprecation warnings
            deprecation_warnings = get_deprecation_warnings(detected_version)
            for warning in deprecation_warnings:
                compatibility_issues.append(f"Deprecation: {warning}")
            
            logger.info(f"Migration available: {detected_version_str} -> {target_version}")
            return False, detected_version_str, compatibility_issues
        
        # No migration path available
        compatibility_issues.append(
            f"No migration path available from {detected_version_str} to {target_version}"
        )
        compatibility_issues.append(
            "Manual configuration update required"
        )
        logger.error(f"No migration path: {detected_version_str} -> {target_version}")
        return False, detected_version_str, compatibility_issues
        
    except Exception as e:
        logger.error(f"Schema compatibility validation failed: {e}")
        compatibility_issues.append(f"Compatibility check error: {str(e)}")
        return False, "unknown", compatibility_issues


def validate_config_with_version_fallback(
    config_data: Dict[str, Any],
    strict_mode: bool = False
) -> Tuple[bool, ConfigVersion, List[str], Optional[Dict[str, Any]]]:
    """
    Validate configuration with automatic version detection and fallback handling.
    
    This function implements the version-aware validation strategy defined in Section 4.6.1.1,
    automatically detecting configuration version and applying appropriate validation rules
    with graceful fallback for missing or invalid version fields.
    
    Args:
        config_data: Configuration dictionary to validate
        strict_mode: If True, require explicit version field and strict validation
        
    Returns:
        Tuple[bool, ConfigVersion, List[str], Optional[Dict[str, Any]]]: 
            (is_valid, detected_version, validation_errors, sanitized_config)
            
    Raises:
        TypeError: If config_data is not a dictionary
        
    Example:
        >>> config = {"project": {}, "experiments": {}}
        >>> valid, version, errors, sanitized = validate_config_with_version_fallback(config)
        >>> assert version in [ConfigVersion.V0_1_0, ConfigVersion.V0_2_0, ConfigVersion.V1_0_0]
    """
    logger.debug(f"Validating configuration with version fallback (strict_mode={strict_mode})")
    
    # Input validation
    if not isinstance(config_data, dict):
        logger.error(f"Configuration must be dictionary, got {type(config_data)}")
        raise TypeError(f"Configuration data must be dictionary, got {type(config_data)}")
    
    validation_errors = []
    
    try:
        # Attempt automatic version detection
        try:
            detected_version = detect_config_version(config_data)
            logger.info(f"Automatically detected configuration version: {detected_version.value}")
        except ValueError as e:
            if strict_mode:
                logger.error(f"Version detection failed in strict mode: {e}")
                validation_errors.append(f"Version detection failed: {str(e)}")
                return False, ConfigVersion.V0_1_0, validation_errors, None
            else:
                # Fallback to legacy v0.1.0 in non-strict mode
                logger.warning(f"Version detection failed, falling back to v0.1.0: {e}")
                detected_version = ConfigVersion.V0_1_0
                validation_errors.append(f"Version detection fallback applied: {str(e)}")
        
        # Apply version-specific validation
        if detected_version == ConfigVersion.V0_1_0:
            is_valid, version_errors, sanitized_config = validate_v0_1_0_config(config_data)
            validation_errors.extend(version_errors)
            
        elif detected_version == ConfigVersion.V0_2_0:
            is_valid, version_errors, sanitized_config = validate_v0_2_0_config(config_data)
            validation_errors.extend(version_errors)
            
        elif detected_version == ConfigVersion.V1_0_0:
            # Use Pydantic validation for v1.0.0
            is_valid, version_errors, sanitized_config = _validate_v1_0_0_config(config_data)
            validation_errors.extend(version_errors)
            
        else:
            logger.error(f"Unsupported configuration version: {detected_version}")
            validation_errors.append(f"Unsupported version: {detected_version}")
            return False, detected_version, validation_errors, None
        
        # Add version awareness to sanitized config
        if is_valid and sanitized_config is not None:
            if 'schema_version' not in sanitized_config:
                sanitized_config['schema_version'] = detected_version.value
                logger.debug(f"Added schema_version field: {detected_version.value}")
        
        if is_valid:
            logger.info(f"Configuration validation passed for version {detected_version.value}")
        else:
            logger.warning(f"Configuration validation failed for version {detected_version.value}")
        
        return is_valid, detected_version, validation_errors, sanitized_config
        
    except Exception as e:
        logger.error(f"Unexpected error during version fallback validation: {e}")
        validation_errors.append(f"Validation error: {str(e)}")
        return False, ConfigVersion.V0_1_0, validation_errors, None


def validate_config_with_migration(
    config_data: Dict[str, Any],
    target_version: str = CURRENT_VERSION,
    auto_migrate: bool = True
) -> Tuple[bool, str, List[str], Optional[Dict[str, Any]]]:
    """
    Validate configuration with optional automatic migration to target version.
    
    This function implements the migration validation workflow defined in Section 0.3.1,
    combining version detection, compatibility checking, and automatic migration execution
    with comprehensive error reporting and audit trail generation.
    
    Args:
        config_data: Configuration dictionary to validate and potentially migrate
        target_version: Target schema version for migration
        auto_migrate: If True, automatically apply migrations when available
        
    Returns:
        Tuple[bool, str, List[str], Optional[Dict[str, Any]]]: 
            (is_valid, final_version, messages, migrated_config)
            
    Raises:
        TypeError: If config_data is not a dictionary
        ValueError: If target_version is not a valid semantic version
        
    Example:
        >>> legacy_config = {"project": {}, "experiments": {}}
        >>> valid, version, messages, migrated = validate_config_with_migration(legacy_config)
        >>> assert version == "1.0.0"  # Migrated to current version
    """
    logger.debug(f"Validating configuration with migration to {target_version}")
    
    # Input validation
    if not isinstance(config_data, dict):
        logger.error(f"Configuration must be dictionary, got {type(config_data)}")
        raise TypeError(f"Configuration data must be dictionary, got {type(config_data)}")
    
    try:
        Version(target_version)  # Validate target version format
    except Exception as e:
        logger.error(f"Invalid target version format '{target_version}': {e}")
        raise ValueError(f"Invalid semantic version format: {target_version}") from e
    
    messages = []
    
    try:
        # Step 1: Detect current version and check compatibility
        is_compatible, detected_version, compatibility_issues = validate_schema_compatibility(
            config_data, target_version
        )
        
        messages.extend(compatibility_issues)
        
        # Step 2: If already compatible, validate directly
        if is_compatible:
            logger.info(f"Configuration already compatible with {target_version}")
            
            # Validate using version-specific validator
            is_valid, _, validation_errors, sanitized_config = validate_config_with_version_fallback(
                config_data, strict_mode=True
            )
            
            messages.extend(validation_errors)
            return is_valid, detected_version, messages, sanitized_config
        
        # Step 3: Check if migration is needed and available
        from ..migration.versions import is_migration_available, get_migration_path
        
        if not is_migration_available(detected_version, target_version):
            logger.error(f"Migration not available: {detected_version} -> {target_version}")
            messages.append(f"Migration not supported from {detected_version} to {target_version}")
            return False, detected_version, messages, None
        
        # Step 4: Execute migration if auto_migrate is enabled
        if auto_migrate:
            try:
                # Import migration functionality
                from ..migration.migrators import migrate_config
                
                migration_path = get_migration_path(detected_version, target_version)
                logger.info(f"Executing migration path: {' -> '.join(str(v) for v in migration_path)}")
                
                migrated_config = migrate_config(config_data, target_version)
                messages.append(f"Successfully migrated from {detected_version} to {target_version}")
                
                # Validate migrated configuration
                is_valid, _, validation_errors, final_config = validate_config_with_version_fallback(
                    migrated_config, strict_mode=True
                )
                
                messages.extend(validation_errors)
                
                if is_valid:
                    logger.info(f"Migration validation passed for {target_version}")
                    return True, target_version, messages, final_config
                else:
                    logger.error(f"Migration validation failed for {target_version}")
                    messages.append("Migrated configuration failed validation")
                    return False, target_version, messages, None
                
            except ImportError as e:
                logger.error(f"Migration module not available: {e}")
                messages.append(f"Migration functionality not available: {str(e)}")
                return False, detected_version, messages, None
            
            except Exception as e:
                logger.error(f"Migration execution failed: {e}")
                messages.append(f"Migration failed: {str(e)}")
                return False, detected_version, messages, None
        
        else:
            # Auto-migration disabled - return compatibility information
            messages.append(f"Migration available but auto_migrate=False")
            messages.append(f"Manual migration required from {detected_version} to {target_version}")
            logger.info(f"Migration available but not executed: auto_migrate=False")
            return False, detected_version, messages, None
            
    except Exception as e:
        logger.error(f"Unexpected error during migration validation: {e}")
        messages.append(f"Migration validation error: {str(e)}")
        return False, "unknown", messages, None


def validate_version_format(version_string: str) -> Tuple[bool, str, Optional[Version]]:
    """
    Validate version string format against semantic versioning requirements.
    
    This function ensures version strings conform to semantic versioning specification
    and are compatible with the FlyRigLoader version management system. It provides
    detailed validation messages for debugging and user guidance.
    
    Args:
        version_string: Version string to validate
        
    Returns:
        Tuple[bool, str, Optional[Version]]: (is_valid, message, parsed_version)
        
    Raises:
        TypeError: If version_string is not a string
        
    Example:
        >>> valid, message, version = validate_version_format("1.0.0")
        >>> assert valid == True
        >>> assert version.major == 1
    """
    logger.debug(f"Validating version format: {version_string}")
    
    # Input type validation
    if not isinstance(version_string, str):
        logger.error(f"Version must be string, got {type(version_string)}")
        raise TypeError(f"Version string must be string, got {type(version_string)}")
    
    # Check for empty or whitespace-only version
    if not version_string.strip():
        logger.warning("Empty or whitespace-only version string")
        return False, "Version string cannot be empty or whitespace-only", None
    
    version_str = version_string.strip()
    
    try:
        # Parse using semantic_version library
        parsed_version = Version(version_str)
        
        # Validate against supported FlyRigLoader versions
        from ..migration.versions import ConfigVersion
        
        supported_versions = [v.value for v in ConfigVersion]
        
        if version_str in supported_versions:
            logger.info(f"Version {version_str} is supported by FlyRigLoader")
            return True, f"Valid and supported version: {version_str}", parsed_version
        else:
            logger.warning(f"Version {version_str} is valid but not officially supported")
            return True, f"Valid semantic version but not officially supported: {version_str}", parsed_version
            
    except Exception as e:
        logger.error(f"Invalid semantic version format '{version_str}': {e}")
        
        # Provide helpful error messages for common mistakes
        error_message = f"Invalid semantic version format: {str(e)}"
        
        if not re.match(r'^\d+\.\d+\.\d+', version_str):
            error_message += ". Expected format: MAJOR.MINOR.PATCH (e.g., '1.0.0')"
        
        if re.search(r'[^0-9\.\-\+\w]', version_str):
            error_message += ". Contains invalid characters"
        
        return False, error_message, None


def validate_legacy_config_format(
    config_data: Dict[str, Any],
    expected_version: Optional[str] = None
) -> Tuple[bool, str, List[str]]:
    """
    Validate legacy configuration format with comprehensive format checking.
    
    This function provides specialized validation for legacy configuration formats
    (v0.1.0 and v0.2.0), ensuring structural integrity while identifying upgrade
    opportunities and potential compatibility issues.
    
    Args:
        config_data: Configuration dictionary in legacy format
        expected_version: Expected legacy version for validation (optional)
        
    Returns:
        Tuple[bool, str, List[str]]: (is_valid, detected_format, validation_messages)
        
    Raises:
        TypeError: If config_data is not a dictionary
        
    Example:
        >>> legacy_config = {"project": {"directories": {}}, "experiments": {}}
        >>> valid, format_type, messages = validate_legacy_config_format(legacy_config)
        >>> assert format_type in ["v0.1.0", "v0.2.0"]
    """
    logger.debug(f"Validating legacy configuration format (expected: {expected_version})")
    
    # Input validation
    if not isinstance(config_data, dict):
        logger.error(f"Configuration must be dictionary, got {type(config_data)}")
        raise TypeError(f"Configuration data must be dictionary, got {type(config_data)}")
    
    validation_messages = []
    
    try:
        # Detect legacy format version
        has_project = 'project' in config_data and isinstance(config_data['project'], dict)
        has_datasets = 'datasets' in config_data and isinstance(config_data['datasets'], dict)
        has_experiments = 'experiments' in config_data and isinstance(config_data['experiments'], dict)
        has_schema_version = 'schema_version' in config_data
        
        logger.debug(f"Legacy format analysis - project: {has_project}, datasets: {has_datasets}, "
                    f"experiments: {has_experiments}, schema_version: {has_schema_version}")
        
        # If explicit schema_version present, it's not legacy format
        if has_schema_version:
            schema_version = config_data['schema_version']
            logger.info(f"Configuration has explicit schema_version: {schema_version}")
            validation_messages.append(f"Not legacy format - contains schema_version: {schema_version}")
            return False, "modern", validation_messages
        
        # Determine legacy format version
        detected_format = None
        
        if has_project and has_datasets and has_experiments:
            detected_format = "v0.2.0"
            logger.info("Detected legacy v0.2.0 format (project + datasets + experiments)")
        elif has_project and has_experiments and not has_datasets:
            detected_format = "v0.1.0"
            logger.info("Detected legacy v0.1.0 format (project + experiments, no datasets)")
        else:
            validation_messages.append("Unrecognized configuration structure")
            if not has_project:
                validation_messages.append("Missing required 'project' section")
            if not has_experiments:
                validation_messages.append("Missing required 'experiments' section")
            logger.warning("Configuration structure does not match known legacy formats")
            return False, "unknown", validation_messages
        
        # Validate against expected version if provided
        if expected_version and detected_format != expected_version:
            validation_messages.append(
                f"Format mismatch: expected {expected_version}, detected {detected_format}"
            )
            logger.warning(f"Legacy format mismatch: expected {expected_version}, got {detected_format}")
            return False, detected_format, validation_messages
        
        # Perform format-specific validation
        if detected_format == "v0.1.0":
            is_valid, format_errors, _ = validate_v0_1_0_config(config_data)
            validation_messages.extend(format_errors)
        elif detected_format == "v0.2.0":
            is_valid, format_errors, _ = validate_v0_2_0_config(config_data)
            validation_messages.extend(format_errors)
        else:
            validation_messages.append(f"Unsupported legacy format: {detected_format}")
            return False, detected_format, validation_messages
        
        # Add legacy format warnings
        validation_messages.append(f"Legacy format {detected_format} detected")
        validation_messages.append(f"Consider upgrading to v1.0.0 for enhanced features and validation")
        
        if is_valid:
            logger.info(f"Legacy {detected_format} format validation passed")
        else:
            logger.warning(f"Legacy {detected_format} format validation failed")
        
        return is_valid, detected_format, validation_messages
        
    except Exception as e:
        logger.error(f"Unexpected error during legacy format validation: {e}")
        validation_messages.append(f"Legacy format validation error: {str(e)}")
        return False, "error", validation_messages


# Private helper functions for version-specific validation

def _validate_v0_1_0_project_section(project_data: Dict[str, Any]) -> List[str]:
    """Validate v0.1.0 project section structure."""
    errors = []
    
    if not isinstance(project_data, dict):
        errors.append("Project section must be a dictionary")
        return errors
    
    # Basic directory validation for v0.1.0
    if 'directories' in project_data:
        directories = project_data['directories']
        if not isinstance(directories, dict):
            errors.append("Project directories must be a dictionary")
        else:
            # Validate directory paths with security checks
            for dir_name, dir_path in directories.items():
                if dir_path is not None:
                    try:
                        # Import locally to avoid circular imports
                        from ..config.validators import path_traversal_protection
                        path_traversal_protection(str(dir_path))
                    except (ValueError, PermissionError) as e:
                        errors.append(f"Invalid directory path '{dir_name}': {str(e)}")
    
    return errors


def _validate_v0_2_0_project_section(project_data: Dict[str, Any]) -> List[str]:
    """Validate v0.2.0 project section structure with enhanced checks."""
    errors = []
    
    # Include basic v0.1.0 validation
    errors.extend(_validate_v0_1_0_project_section(project_data))
    
    # Additional v0.2.0 specific validation
    if 'ignore_substrings' in project_data:
        ignore_patterns = project_data['ignore_substrings']
        if ignore_patterns is not None and not isinstance(ignore_patterns, list):
            errors.append("Project ignore_substrings must be a list")
    
    if 'extraction_patterns' in project_data:
        patterns = project_data['extraction_patterns']
        if patterns is not None:
            if not isinstance(patterns, list):
                errors.append("Project extraction_patterns must be a list")
            else:
                for pattern in patterns:
                    if not isinstance(pattern, str):
                        errors.append("Extraction patterns must be strings")
                        continue
                    try:
                        re.compile(pattern)
                    except re.error as e:
                        errors.append(f"Invalid extraction pattern '{pattern}': {str(e)}")
    
    return errors


def _validate_v0_1_0_experiments_section(experiments_data: Dict[str, Any]) -> List[str]:
    """Validate v0.1.0 experiments section structure."""
    errors = []
    
    if not isinstance(experiments_data, dict):
        errors.append("Experiments section must be a dictionary")
        return errors
    
    for exp_name, exp_config in experiments_data.items():
        if not isinstance(exp_config, dict):
            errors.append(f"Experiment '{exp_name}' configuration must be a dictionary")
            continue
        
        # Basic experiment validation
        if 'date_range' in exp_config:
            date_range = exp_config['date_range']
            if isinstance(date_range, list) and len(date_range) >= 2:
                for date_str in date_range[:2]:  # Validate start and end dates
                    try:
                        # Import locally to avoid circular imports
                        from ..config.validators import date_format_validator
                        date_format_validator(str(date_str))
                    except (ValueError, TypeError) as e:
                        errors.append(f"Invalid date in experiment '{exp_name}': {str(e)}")
    
    return errors


def _validate_v0_2_0_datasets_section(datasets_data: Dict[str, Any]) -> List[str]:
    """Validate v0.2.0 datasets section structure."""
    errors = []
    
    if not isinstance(datasets_data, dict):
        errors.append("Datasets section must be a dictionary")
        return errors
    
    for dataset_name, dataset_config in datasets_data.items():
        if not isinstance(dataset_config, dict):
            errors.append(f"Dataset '{dataset_name}' configuration must be a dictionary")
            continue
        
        # Basic dataset validation
        if 'rig' not in dataset_config:
            errors.append(f"Dataset '{dataset_name}' missing required 'rig' field")
        elif not isinstance(dataset_config['rig'], str):
            errors.append(f"Dataset '{dataset_name}' rig must be a string")
        
        if 'dates_vials' in dataset_config:
            dates_vials = dataset_config['dates_vials']
            if not isinstance(dates_vials, dict):
                errors.append(f"Dataset '{dataset_name}' dates_vials must be a dictionary")
            else:
                for date_str, vials in dates_vials.items():
                    try:
                        # Import locally to avoid circular imports
                        from ..config.validators import date_format_validator
                        date_format_validator(date_str)
                    except (ValueError, TypeError) as e:
                        errors.append(f"Invalid date in dataset '{dataset_name}': {str(e)}")
                    
                    if not isinstance(vials, list):
                        errors.append(f"Vials for date '{date_str}' must be a list")
    
    return errors


def _validate_v0_2_0_experiments_section(
    experiments_data: Dict[str, Any], 
    available_datasets: List[str]
) -> List[str]:
    """Validate v0.2.0 experiments section with dataset references."""
    errors = []
    
    # Include basic v0.1.0 validation
    errors.extend(_validate_v0_1_0_experiments_section(experiments_data))
    
    # Additional v0.2.0 validation for dataset references
    for exp_name, exp_config in experiments_data.items():
        if not isinstance(exp_config, dict):
            continue
        
        if 'datasets' in exp_config:
            exp_datasets = exp_config['datasets']
            if not isinstance(exp_datasets, list):
                errors.append(f"Experiment '{exp_name}' datasets must be a list")
            else:
                for dataset_ref in exp_datasets:
                    if dataset_ref not in available_datasets:
                        errors.append(f"Experiment '{exp_name}' references unknown dataset: {dataset_ref}")
    
    return errors


def _validate_v1_0_0_config(config_data: Dict[str, Any]) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    """Validate v1.0.0 configuration using Pydantic models."""
    errors = []
    
    try:
        # Extract sections for Pydantic validation
        project_data = config_data.get('project', {})
        datasets_data = config_data.get('datasets', {})
        experiments_data = config_data.get('experiments', {})
        
        # Validate project section
        try:
            # Import locally to avoid circular imports
            from ..config.models import ProjectConfig
            project_config = ProjectConfig(**project_data)
            logger.debug("Project section validated successfully with Pydantic")
        except ValidationError as e:
            for error in e.errors():
                errors.append(f"Project validation error: {error['msg']} at {error['loc']}")
        
        # Validate datasets section
        for dataset_name, dataset_config in datasets_data.items():
            try:
                # Import locally to avoid circular imports
                from ..config.models import DatasetConfig
                DatasetConfig(**dataset_config)
                logger.debug(f"Dataset '{dataset_name}' validated successfully with Pydantic")
            except ValidationError as e:
                for error in e.errors():
                    errors.append(f"Dataset '{dataset_name}' validation error: {error['msg']} at {error['loc']}")
        
        # Validate experiments section
        for exp_name, exp_config in experiments_data.items():
            try:
                # Import locally to avoid circular imports
                from ..config.models import ExperimentConfig
                ExperimentConfig(**exp_config)
                logger.debug(f"Experiment '{exp_name}' validated successfully with Pydantic")
            except ValidationError as e:
                for error in e.errors():
                    errors.append(f"Experiment '{exp_name}' validation error: {error['msg']} at {error['loc']}")
        
        is_valid = len(errors) == 0
        sanitized_config = config_data.copy() if is_valid else None
        
        if is_valid:
            sanitized_config['schema_version'] = '1.0.0'
        
        return is_valid, errors, sanitized_config
        
    except Exception as e:
        logger.error(f"Unexpected error during v1.0.0 validation: {e}")
        errors.append(f"Pydantic validation error: {str(e)}")
        return False, errors, None


# Module initialization
logger.info("Version-specific validation module initialized")