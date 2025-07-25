"""
Package initialization for FlyRigLoader migration infrastructure.

This module provides centralized access to the comprehensive version management system for 
FlyRigLoader configuration schema evolution. It exposes the public API for configuration 
version migration including the ConfigMigrator class, ConfigVersion enum, version constants, 
and all validation utilities required for automatic schema version detection and seamless 
upgrades between configuration versions.

The migration infrastructure implements the technical strategy defined in Section 0.3.1 
of the technical specification, providing:

- Automatic version detection and compatibility validation
- Zero-breaking-change configuration upgrades with comprehensive audit trails
- Pydantic-based schema validation with progressive enhancement capabilities
- Thread-safe migration execution for concurrent research computing environments
- Integration with the broader FlyRigLoader ecosystem for enterprise-grade reliability

Key Components Exposed:
- ConfigMigrator: Main migration engine with registry and audit capabilities
- ConfigVersion: Enumeration of supported configuration schema versions
- MigrationReport: Comprehensive audit trail and logging infrastructure  
- Version detection and validation utilities for automated migration workflows
- Constants and compatibility matrices for version management operations

Usage Examples:
    Basic configuration migration:
    >>> from flyrigloader.migration import ConfigMigrator, ConfigVersion
    >>> migrator = ConfigMigrator()
    >>> config_dict = {"project": {"directories": {}}, "experiments": {}}
    >>> migrated_config, report = migrator.migrate(config_dict, "0.1.0", "1.0.0")
    >>> print(f"Migration successful: {len(report.errors) == 0}")
    
    Version detection and validation:
    >>> from flyrigloader.migration import detect_config_version, validate_schema_compatibility
    >>> version = detect_config_version(config_dict)
    >>> compatible, detected, issues = validate_schema_compatibility(config_dict)
    >>> print(f"Detected version: {detected}, compatible: {compatible}")
    
    Configuration with automatic migration:
    >>> from flyrigloader.migration import validate_config_with_migration
    >>> valid, final_version, messages, migrated = validate_config_with_migration(config_dict)
    >>> print(f"Final version: {final_version}, valid: {valid}")

Architecture Integration:
This module serves as the single entry point for all version management operations,
implementing the centralized API strategy defined in Section 5.3.6 of the technical
specification. It ensures consistent access patterns while maintaining backward
compatibility through the LegacyConfigAdapter and deprecation warning systems.

The migration infrastructure supports the broader FlyRigLoader refactoring initiative
by enabling incremental adoption of enhanced features while preserving research
workflow continuity and institutional compliance requirements.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

# Set up logger for migration package operations
logger = logging.getLogger(__name__)

# Import version constants and enums from versions module
from flyrigloader.migration.versions import (
    CURRENT_VERSION,
    ConfigVersion,
    COMPATIBILITY_MATRIX,
    detect_config_version,
    get_migration_path,
    is_migration_available,
    get_version_info,
    validate_version_compatibility,
    get_deprecation_warnings,
    compare_versions
)

# Import migration execution engine from migrators module
from flyrigloader.migration.migrators import (
    ConfigMigrator,
    MigrationReport,
    migrate_v0_1_to_v1_0,
    migrate_v0_2_to_v1_0
)

# Import comprehensive validation functions from validators module
from flyrigloader.migration.validators import (
    validate_schema_compatibility,
    validate_config_with_version_fallback,
    validate_config_with_migration,
    validate_version_format,
    validate_legacy_config_format,
    validate_v0_1_0_config,
    validate_v0_2_0_config
)

# Additional utility functions for comprehensive API coverage
def create_migration_report(from_version: str, to_version: str) -> MigrationReport:
    """
    Create a new migration report for tracking configuration transformations.
    
    This function provides a convenient factory method for creating MigrationReport
    instances with proper initialization and timestamp handling. It's designed for
    use in custom migration workflows and testing scenarios.
    
    Args:
        from_version: Source configuration schema version
        to_version: Target configuration schema version
        
    Returns:
        MigrationReport: Initialized report ready for migration tracking
        
    Example:
        >>> report = create_migration_report("0.1.0", "1.0.0")
        >>> report.add_warning("Legacy format detected")
        >>> report.applied_migrations.append("migrate_v0_1_to_v1_0")
        >>> print(report.to_dict())
    """
    logger.debug(f"Creating migration report: {from_version} -> {to_version}")
    return MigrationReport(from_version, to_version)


def get_supported_versions() -> List[str]:
    """
    Get list of all supported configuration schema versions.
    
    This function returns a comprehensive list of all configuration versions
    supported by the current FlyRigLoader migration system, useful for
    validation workflows and user interface generation.
    
    Returns:
        List[str]: All supported version strings in chronological order
        
    Example:
        >>> versions = get_supported_versions()
        >>> print(f"Supported versions: {', '.join(versions)}")
        ['0.1.0', '0.2.0', '1.0.0']
    """
    logger.debug("Retrieving list of supported configuration versions")
    return [version.value for version in ConfigVersion]


def is_current_version(version: Union[str, ConfigVersion]) -> bool:
    """
    Check if the provided version is the current schema version.
    
    This utility function provides a convenient way to determine if a
    configuration version represents the current system version, useful
    for migration decision logic and compatibility checking.
    
    Args:
        version: Configuration version to check (string or enum)
        
    Returns:
        bool: True if version matches current system version
        
    Example:
        >>> assert is_current_version("1.0.0") == True
        >>> assert is_current_version("0.1.0") == False
    """
    if isinstance(version, ConfigVersion):
        version_str = version.value
    else:
        version_str = str(version)
    
    result = version_str == CURRENT_VERSION
    logger.debug(f"Version {version_str} is current: {result}")
    return result


def get_migration_summary(from_version: str, to_version: str) -> Dict[str, Any]:
    """
    Get comprehensive migration summary including path, compatibility, and warnings.
    
    This function provides a detailed analysis of a potential migration, including
    the migration path, compatibility information, deprecation warnings, and
    estimated complexity. It's designed for migration planning and user guidance.
    
    Args:
        from_version: Source configuration version
        to_version: Target configuration version
        
    Returns:
        Dict[str, Any]: Comprehensive migration analysis
        
    Raises:
        ValueError: If migration is not possible between the specified versions
        
    Example:
        >>> summary = get_migration_summary("0.1.0", "1.0.0")
        >>> print(f"Migration steps: {summary['migration_steps']}")
        >>> print(f"Warnings: {len(summary['deprecation_warnings'])}")
    """
    logger.debug(f"Generating migration summary: {from_version} -> {to_version}")
    
    try:
        # Validate versions
        from_ver = ConfigVersion.from_string(from_version)
        to_ver = ConfigVersion.from_string(to_version)
        
        # Check migration availability
        if not is_migration_available(from_version, to_version):
            raise ValueError(f"Migration not available from {from_version} to {to_version}")
        
        # Calculate migration path
        migration_path = get_migration_path(from_version, to_version)
        migration_steps = len(migration_path) - 1
        
        # Get deprecation warnings for source version
        deprecation_warnings = get_deprecation_warnings(from_ver)
        
        # Get version information
        from_info = get_version_info(from_version)
        to_info = get_version_info(to_version)
        
        # Determine migration complexity
        complexity = "simple" if migration_steps == 1 else "complex"
        if migration_steps > 2:
            complexity = "advanced"
        
        summary = {
            "migration_path": [str(v) for v in migration_path],
            "migration_steps": migration_steps,
            "complexity": complexity,
            "from_version_info": from_info,
            "to_version_info": to_info,
            "deprecation_warnings": deprecation_warnings,
            "breaking_changes": to_info.get("breaking_changes", []),
            "new_features": to_info.get("features", []),
            "estimated_time": "immediate" if migration_steps <= 2 else "moderate",
            "rollback_available": migration_steps <= 2,
            "validation_required": True
        }
        
        logger.info(f"Migration summary generated: {migration_steps} steps, complexity: {complexity}")
        return summary
        
    except Exception as e:
        logger.error(f"Failed to generate migration summary: {e}")
        raise ValueError(f"Cannot generate migration summary: {e}") from e


def validate_configuration_compatibility(
    config_data: Dict[str, Any], 
    target_versions: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Validate configuration compatibility against multiple target versions.
    
    This function performs comprehensive compatibility analysis against multiple
    target versions simultaneously, providing detailed results for migration
    planning and compatibility assessment workflows.
    
    Args:
        config_data: Configuration dictionary to analyze
        target_versions: List of versions to check compatibility against
                        (defaults to all supported versions)
        
    Returns:
        Dict[str, Dict[str, Any]]: Compatibility results keyed by target version
        
    Example:
        >>> config = {"project": {}, "experiments": {}}
        >>> results = validate_configuration_compatibility(config)
        >>> for version, result in results.items():
        ...     print(f"Version {version}: compatible={result['compatible']}")
    """
    if target_versions is None:
        target_versions = get_supported_versions()
    
    logger.debug(f"Validating configuration compatibility against {len(target_versions)} versions")
    
    results = {}
    
    for target_version in target_versions:
        try:
            is_compatible, detected_version, issues = validate_schema_compatibility(
                config_data, target_version
            )
            
            # Get additional migration information if not compatible
            migration_info = {}
            if not is_compatible:
                try:
                    if is_migration_available(detected_version, target_version):
                        migration_path = get_migration_path(detected_version, target_version)
                        migration_info = {
                            "migration_available": True,
                            "migration_path": [str(v) for v in migration_path],
                            "migration_steps": len(migration_path) - 1
                        }
                    else:
                        migration_info = {
                            "migration_available": False,
                            "manual_update_required": True
                        }
                except Exception as e:
                    migration_info = {
                        "migration_available": False,
                        "error": str(e)
                    }
            
            results[target_version] = {
                "compatible": is_compatible,
                "detected_version": detected_version,
                "compatibility_issues": issues,
                "migration_info": migration_info
            }
            
        except Exception as e:
            logger.warning(f"Compatibility check failed for version {target_version}: {e}")
            results[target_version] = {
                "compatible": False,
                "detected_version": "unknown",
                "compatibility_issues": [f"Compatibility check error: {str(e)}"],
                "migration_info": {"error": str(e)}
            }
    
    logger.info(f"Compatibility validation completed for {len(results)} versions")
    return results


# Enhanced public API with comprehensive coverage
__all__ = [
    # Core migration engine and reporting
    "ConfigMigrator",
    "MigrationReport",
    
    # Version management constants and enum
    "ConfigVersion", 
    "CURRENT_VERSION",
    "COMPATIBILITY_MATRIX",
    
    # Version detection and analysis functions
    "detect_config_version",
    "get_migration_path",
    "is_migration_available",
    "get_version_info",
    "validate_version_compatibility",
    "get_deprecation_warnings",
    "compare_versions",
    
    # Comprehensive validation functions
    "validate_schema_compatibility",
    "validate_config_with_version_fallback", 
    "validate_config_with_migration",
    "validate_version_format",
    "validate_legacy_config_format",
    
    # Version-specific migration functions
    "migrate_v0_1_to_v1_0",
    "migrate_v0_2_to_v1_0",
    
    # Utility and convenience functions
    "create_migration_report",
    "get_supported_versions",
    "is_current_version",
    "get_migration_summary",
    "validate_configuration_compatibility"
]

# Module initialization logging
logger.info(f"FlyRigLoader migration infrastructure initialized - current version: {CURRENT_VERSION}")
logger.debug(f"Migration package exports {len(__all__)} public functions and classes")