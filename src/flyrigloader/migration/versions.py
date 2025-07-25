"""
Version constants and compatibility matrix for FlyRigLoader configuration schema versioning.

This module provides the centralized version management system for FlyRigLoader configuration
schemas, implementing semantic versioning compliance with automatic migration path determination.
It defines the ConfigVersion enum, version compatibility relationships, and utilities for
version parsing and comparison required by the automatic migration system.

The module serves as the authoritative registry of all supported configuration schema versions
with comprehensive compatibility validation logic for safe system upgrades while preserving
research workflow continuity.
"""

import logging
import re
import warnings
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Tuple, Callable

from semantic_version import Version

# Set up logger for version management operations
logger = logging.getLogger(__name__)


class ConfigVersion(Enum):
    """
    Enumeration of supported FlyRigLoader configuration schema versions.
    
    This enum defines all supported configuration schema versions following semantic
    versioning principles. Each version represents a distinct configuration schema
    with specific validation rules and migration requirements.
    
    Version History:
        V0_1_0: Initial legacy configuration format (dictionary-based)
        V0_2_0: Enhanced legacy format with improved validation
        V1_0_0: Modern Pydantic-based configuration with full feature support
    
    Usage:
        >>> version = ConfigVersion.V1_0_0
        >>> print(version.value)
        1.0.0
    """
    
    V0_1_0 = "0.1.0"
    V0_2_0 = "0.2.0" 
    V1_0_0 = "1.0.0"
    
    def __str__(self) -> str:
        """Return the string representation of the version."""
        return self.value
    
    def to_semantic_version(self) -> Version:
        """
        Convert the enum value to a semantic_version.Version object.
        
        Returns:
            Version: Parsed semantic version object for comparison operations
            
        Example:
            >>> version = ConfigVersion.V1_0_0
            >>> sem_ver = version.to_semantic_version()
            >>> print(sem_ver.major, sem_ver.minor, sem_ver.patch)
            1 0 0
        """
        return Version(self.value)
    
    @classmethod
    def from_string(cls, version_string: str) -> 'ConfigVersion':
        """
        Create a ConfigVersion from a version string.
        
        Args:
            version_string: Version string in semantic version format
            
        Returns:
            ConfigVersion: Matching enum value
            
        Raises:
            ValueError: If version string doesn't match any known version
            
        Example:
            >>> version = ConfigVersion.from_string("1.0.0")
            >>> assert version == ConfigVersion.V1_0_0
        """
        # Normalize version string
        normalized = version_string.strip()
        
        for version in cls:
            if version.value == normalized:
                return version
        
        logger.warning(f"Unknown configuration version: {version_string}")
        raise ValueError(f"Unsupported configuration version: {version_string}")
    
    @classmethod
    def get_latest(cls) -> 'ConfigVersion':
        """
        Get the latest available configuration version.
        
        Returns:
            ConfigVersion: The most recent version enum value
        """
        # Sort versions by semantic version and return the latest
        versions = [(v, v.to_semantic_version()) for v in cls]
        latest = max(versions, key=lambda x: x[1])
        return latest[0]


# Current version constant for the refactored system
CURRENT_VERSION: str = ConfigVersion.V1_0_0.value

# Version detection patterns for automatic schema identification
VERSION_PATTERNS: Dict[str, re.Pattern] = {
    # Explicit schema_version field in YAML/dict configurations
    'explicit_field': re.compile(
        r'schema_version\s*:\s*["\']?(\d+\.\d+\.\d+)["\']?',
        re.MULTILINE | re.IGNORECASE
    ),
    
    # Legacy version indicators based on configuration structure
    'legacy_v0_1': re.compile(
        r'(?=.*project\s*:)(?=.*experiments\s*:)(?!.*schema_version)',
        re.MULTILINE | re.IGNORECASE | re.DOTALL
    ),
    
    'legacy_v0_2': re.compile(
        r'(?=.*project\s*:)(?=.*datasets\s*:)(?=.*experiments\s*:)(?!.*schema_version)',
        re.MULTILINE | re.IGNORECASE | re.DOTALL
    ),
    
    # Pydantic model indicators
    'pydantic_model': re.compile(
        r'(?=.*ProjectConfig)(?=.*DatasetConfig)(?=.*ExperimentConfig)',
        re.MULTILINE | re.DOTALL
    )
}


# Compatibility matrix defining which versions can be migrated to which targets
COMPATIBILITY_MATRIX: Dict[str, Dict[str, Union[bool, Callable[[Dict[str, Any]], Dict[str, Any]]]]] = {
    "0.1.0": {
        "0.2.0": True,  # Can migrate from 0.1.0 to 0.2.0
        "1.0.0": True,  # Can migrate from 0.1.0 to 1.0.0 (via chain)
        "description": "Initial legacy format with basic project/experiments structure",
        "features": ["project", "experiments"],
        "limitations": ["no_datasets", "no_validation", "no_versioning"]
    },
    "0.2.0": {
        "1.0.0": True,  # Can migrate from 0.2.0 to 1.0.0
        "description": "Enhanced legacy format with datasets support",
        "features": ["project", "datasets", "experiments", "basic_validation"],
        "limitations": ["no_pydantic", "no_versioning", "limited_error_handling"]
    },
    "1.0.0": {
        "description": "Modern Pydantic-based configuration with full feature support",
        "features": [
            "pydantic_models", "comprehensive_validation", "builder_patterns",
            "version_management", "kedro_integration", "registry_enforcement"
        ],
        "limitations": [],
        "breaking_changes": ["dictionary_config_deprecated", "new_validation_rules"]
    }
}


def detect_config_version(config_data: Union[Dict[str, Any], str]) -> ConfigVersion:
    """
    Automatically detect the configuration schema version from configuration data.
    
    This function analyzes configuration data to determine the appropriate schema version
    using pattern matching and structural analysis. It supports both dictionary-based
    configurations and raw YAML/text content for comprehensive version detection.
    
    Args:
        config_data: Configuration data as dictionary or raw text string
        
    Returns:
        ConfigVersion: Detected configuration version
        
    Raises:
        ValueError: If version cannot be determined from the data
        
    Example:
        >>> config = {"project": {"directories": {}}, "experiments": {}}
        >>> version = detect_config_version(config)
        >>> assert version == ConfigVersion.V0_1_0
        
        >>> modern_config = {"schema_version": "1.0.0", "project": {}}
        >>> version = detect_config_version(modern_config)
        >>> assert version == ConfigVersion.V1_0_0
    """
    logger.debug("Detecting configuration version from provided data")
    
    # Handle dictionary-based configuration data
    if isinstance(config_data, dict):
        return _detect_version_from_dict(config_data)
    
    # Handle string/text-based configuration data (YAML content)
    elif isinstance(config_data, str):
        return _detect_version_from_string(config_data)
    
    # Handle LegacyConfigAdapter objects
    elif hasattr(config_data, '__class__') and 'LegacyConfigAdapter' in str(config_data.__class__):
        # Extract the underlying data from LegacyConfigAdapter
        if hasattr(config_data, '_data'):
            return _detect_version_from_dict(config_data._data)
        elif hasattr(config_data, 'to_dict'):
            return _detect_version_from_dict(config_data.to_dict())
        else:
            # Fallback: try to access as dict-like object
            config_dict = {}
            try:
                if hasattr(config_data, 'get'):
                    # Try to extract the key data we need for version detection
                    if config_data.get('schema_version'):
                        config_dict['schema_version'] = config_data.get('schema_version')
                    if config_data.get('project'):
                        config_dict['project'] = config_data.get('project')
                    if config_data.get('experiments'):
                        config_dict['experiments'] = config_data.get('experiments')
                    return _detect_version_from_dict(config_dict)
                else:
                    raise ValueError(f"Cannot extract configuration data from LegacyConfigAdapter")
            except Exception as e:
                logger.error(f"Failed to extract data from LegacyConfigAdapter: {e}")
                raise ValueError(f"Cannot detect version from LegacyConfigAdapter: {e}")
    
    else:
        logger.error(f"Unsupported configuration data type: {type(config_data)}")
        raise ValueError(f"Cannot detect version from {type(config_data)} data")


def _detect_version_from_dict(config_dict: Dict[str, Any]) -> ConfigVersion:
    """
    Detect version from dictionary-based configuration data.
    
    Args:
        config_dict: Configuration data as dictionary
        
    Returns:
        ConfigVersion: Detected version based on structure and content analysis
    """
    logger.debug("Analyzing dictionary structure for version detection")
    
    # Check for explicit schema_version field (highest priority)
    if 'schema_version' in config_dict:
        schema_version = config_dict['schema_version']
        logger.info(f"Found explicit schema_version field: {schema_version}")
        
        try:
            return ConfigVersion.from_string(str(schema_version))
        except ValueError as e:
            logger.warning(f"Invalid schema_version value '{schema_version}': {e}")
            # Continue with structural analysis as fallback
    
    # Structural analysis for legacy configurations
    has_project = 'project' in config_dict
    has_datasets = 'datasets' in config_dict  
    has_experiments = 'experiments' in config_dict
    
    logger.debug(f"Configuration structure - project: {has_project}, "
                f"datasets: {has_datasets}, experiments: {has_experiments}")
    
    # Version 0.2.0: Has project, datasets, and experiments
    if has_project and has_datasets and has_experiments:
        logger.info("Detected configuration structure matching version 0.2.0")
        return ConfigVersion.V0_2_0
    
    # Version 0.1.0: Has project and experiments, but no datasets
    elif has_project and has_experiments and not has_datasets:
        logger.info("Detected configuration structure matching version 0.1.0")
        return ConfigVersion.V0_1_0
    
    # Modern configuration without explicit version (assume current)
    elif has_project:
        logger.info("Detected modern configuration structure, assuming current version")
        return ConfigVersion.get_latest()
    
    # Unknown structure
    else:
        logger.error("Unable to determine configuration version from structure")
        raise ValueError("Cannot determine configuration version: unrecognized structure")


def _detect_version_from_string(config_string: str) -> ConfigVersion:
    """
    Detect version from string-based configuration data (e.g., YAML content).
    
    Args:
        config_string: Raw configuration content as string
        
    Returns:
        ConfigVersion: Detected version based on pattern matching
    """
    logger.debug("Analyzing string content for version detection patterns")
    
    # Check for explicit schema_version field pattern
    version_match = VERSION_PATTERNS['explicit_field'].search(config_string)
    if version_match:
        detected_version = version_match.group(1)
        logger.info(f"Found explicit schema_version in content: {detected_version}")
        
        try:
            return ConfigVersion.from_string(detected_version)
        except ValueError as e:
            logger.warning(f"Invalid schema_version in content '{detected_version}': {e}")
    
    # Pattern-based structural analysis
    if VERSION_PATTERNS['legacy_v0_2'].search(config_string):
        logger.info("Detected legacy v0.2.0 pattern in content")
        return ConfigVersion.V0_2_0
    
    elif VERSION_PATTERNS['legacy_v0_1'].search(config_string):
        logger.info("Detected legacy v0.1.0 pattern in content") 
        return ConfigVersion.V0_1_0
    
    elif VERSION_PATTERNS['pydantic_model'].search(config_string):
        logger.info("Detected Pydantic model references, assuming current version")
        return ConfigVersion.get_latest()
    
    # Default fallback
    logger.warning("Unable to detect version from string content, assuming legacy v0.1.0")
    warnings.warn(
        "Configuration version could not be determined, assuming legacy format v0.1.0. "
        "Consider adding explicit 'schema_version' field to your configuration.",
        DeprecationWarning,
        stacklevel=3
    )
    return ConfigVersion.V0_1_0


def get_migration_path(
    from_version: Union[str, ConfigVersion], 
    to_version: Union[str, ConfigVersion]
) -> List[ConfigVersion]:
    """
    Calculate the migration path between two configuration versions.
    
    This function determines the sequence of version upgrades needed to migrate
    from one configuration version to another, supporting both direct migrations
    and multi-step migration chains for complex version transitions.
    
    Args:
        from_version: Source configuration version
        to_version: Target configuration version
        
    Returns:
        List[ConfigVersion]: Ordered list of versions in migration path
        
    Raises:
        ValueError: If migration path is not available or supported
        
    Example:
        >>> path = get_migration_path("0.1.0", "1.0.0")
        >>> assert path == [ConfigVersion.V0_1_0, ConfigVersion.V0_2_0, ConfigVersion.V1_0_0]
    """
    logger.debug(f"Calculating migration path from {from_version} to {to_version}")
    
    # Normalize inputs to ConfigVersion objects
    if isinstance(from_version, str):
        from_version = ConfigVersion.from_string(from_version)
    if isinstance(to_version, str):
        to_version = ConfigVersion.from_string(to_version)
    
    # Same version - no migration needed
    if from_version == to_version:
        logger.info("Source and target versions are identical, no migration needed")
        return [from_version]
    
    # Check if direct migration is available
    from_key = from_version.value
    to_key = to_version.value
    
    if from_key in COMPATIBILITY_MATRIX:
        compat_info = COMPATIBILITY_MATRIX[from_key]
        
        # Direct migration available
        if to_key in compat_info and compat_info[to_key] is True:
            logger.info(f"Direct migration path available: {from_version} -> {to_version}")
            return [from_version, to_version]
    
    # Multi-step migration path calculation
    migration_path = _find_migration_chain(from_version, to_version)
    
    if migration_path:
        logger.info(f"Multi-step migration path found: {' -> '.join(str(v) for v in migration_path)}")
        return migration_path
    
    # No migration path available
    logger.error(f"No migration path available from {from_version} to {to_version}")
    raise ValueError(f"Migration from {from_version} to {to_version} is not supported")


def _find_migration_chain(
    from_version: ConfigVersion, 
    to_version: ConfigVersion
) -> Optional[List[ConfigVersion]]:
    """
    Find a multi-step migration chain using breadth-first search.
    
    Args:
        from_version: Source version
        to_version: Target version
        
    Returns:
        Optional[List[ConfigVersion]]: Migration chain if found, None otherwise
    """
    logger.debug(f"Searching for migration chain: {from_version} -> {to_version}")
    
    # Simple implementation for the defined versions
    # In a more complex system, this would use graph algorithms
    
    all_versions = [ConfigVersion.V0_1_0, ConfigVersion.V0_2_0, ConfigVersion.V1_0_0]
    
    # Define the known migration chains
    if from_version == ConfigVersion.V0_1_0 and to_version == ConfigVersion.V1_0_0:
        # 0.1.0 -> 0.2.0 -> 1.0.0
        return [ConfigVersion.V0_1_0, ConfigVersion.V0_2_0, ConfigVersion.V1_0_0]
    
    # No other multi-step chains currently defined
    return None


def is_migration_available(
    from_version: Union[str, ConfigVersion], 
    to_version: Union[str, ConfigVersion]
) -> bool:
    """
    Check if migration is available between two versions.
    
    Args:
        from_version: Source configuration version
        to_version: Target configuration version
        
    Returns:
        bool: True if migration is possible, False otherwise
        
    Example:
        >>> assert is_migration_available("0.1.0", "1.0.0") == True
        >>> assert is_migration_available("1.0.0", "0.1.0") == False
    """
    logger.debug(f"Checking migration availability: {from_version} -> {to_version}")
    
    try:
        get_migration_path(from_version, to_version)
        return True
    except ValueError:
        return False


def get_version_info(version: Union[str, ConfigVersion]) -> Dict[str, Any]:
    """
    Get comprehensive information about a configuration version.
    
    Args:
        version: Configuration version to query
        
    Returns:
        Dict[str, Any]: Version information including features and limitations
        
    Example:
        >>> info = get_version_info("1.0.0")
        >>> print(info['features'])
        ['pydantic_models', 'comprehensive_validation', ...]
    """
    logger.debug(f"Retrieving version information for {version}")
    
    # Normalize to string for matrix lookup
    if isinstance(version, ConfigVersion):
        version_key = version.value
    else:
        version_key = str(version)
    
    if version_key not in COMPATIBILITY_MATRIX:
        logger.warning(f"No information available for version {version}")
        return {
            "version": version_key,
            "description": "Unknown version",
            "features": [],
            "limitations": ["unknown_version"],
            "breaking_changes": []
        }
    
    version_info = COMPATIBILITY_MATRIX[version_key].copy()
    version_info["version"] = version_key
    
    # Remove migration flags from info
    for key in list(version_info.keys()):
        if key not in ["description", "features", "limitations", "breaking_changes", "version"]:
            if not isinstance(version_info[key], (list, str)):
                del version_info[key]
    
    logger.debug(f"Retrieved information for version {version_key}")
    return version_info


def validate_version_compatibility(config_version: str, system_version: str = CURRENT_VERSION) -> Tuple[bool, str]:
    """
    Validate if a configuration version is compatible with the current system.
    
    Args:
        config_version: Version of the configuration to validate
        system_version: Current system version (defaults to CURRENT_VERSION)
        
    Returns:
        Tuple[bool, str]: (is_compatible, message) indicating compatibility status
        
    Example:
        >>> compatible, msg = validate_version_compatibility("0.1.0")
        >>> if not compatible:
        ...     print(f"Migration needed: {msg}")
    """
    logger.debug(f"Validating compatibility: config v{config_version} with system v{system_version}")
    
    try:
        config_ver = ConfigVersion.from_string(config_version)
        system_ver = ConfigVersion.from_string(system_version)
        
        # Same version - fully compatible
        if config_ver == system_ver:
            return True, f"Configuration version {config_version} matches system version"
        
        # Check if migration is available
        if is_migration_available(config_ver, system_ver):
            return False, f"Configuration version {config_version} can be migrated to {system_version}"
        
        # Check if configuration is newer than system
        config_sem = config_ver.to_semantic_version()
        system_sem = system_ver.to_semantic_version()
        
        if config_sem > system_sem:
            return False, f"Configuration version {config_version} is newer than system version {system_version}"
        
        # No migration path available
        return False, f"Configuration version {config_version} is incompatible with system version {system_version}"
    
    except ValueError as e:
        logger.error(f"Version validation failed: {e}")
        return False, f"Invalid version format: {e}"


def get_deprecation_warnings(version: Union[str, ConfigVersion]) -> List[str]:
    """
    Get deprecation warnings for a specific configuration version.
    
    Args:
        version: Configuration version to check for deprecations
        
    Returns:
        List[str]: List of deprecation warning messages
        
    Example:
        >>> warnings = get_deprecation_warnings("0.1.0")
        >>> for warning in warnings:
        ...     print(f"Warning: {warning}")
    """
    logger.debug(f"Retrieving deprecation warnings for version {version}")
    
    # Normalize to ConfigVersion
    if isinstance(version, str):
        try:
            version = ConfigVersion.from_string(version)
        except ValueError:
            return [f"Unknown version {version} - consider upgrading to current version"]
    
    warnings_list = []
    
    # Version-specific deprecation warnings
    if version == ConfigVersion.V0_1_0:
        warnings_list.extend([
            "Dictionary-based configuration format is deprecated",
            "Missing datasets section limits experimental flexibility",
            "No validation - configuration errors may cause runtime failures",
            "Consider migrating to Pydantic-based configuration (v1.0.0)"
        ])
    
    elif version == ConfigVersion.V0_2_0:
        warnings_list.extend([
            "Legacy configuration format will be removed in future versions",
            "Limited error handling compared to modern validation",
            "No Kedro integration support",
            "Upgrade to v1.0.0 recommended for full feature support"
        ])
    
    elif version == ConfigVersion.V1_0_0:
        # Current version - no deprecation warnings
        pass
    
    logger.debug(f"Found {len(warnings_list)} deprecation warnings for version {version}")
    return warnings_list


# Version comparison utilities for external use
def compare_versions(version1: Union[str, ConfigVersion], version2: Union[str, ConfigVersion]) -> int:
    """
    Compare two configuration versions using semantic versioning rules.
    
    Args:
        version1: First version to compare
        version2: Second version to compare
        
    Returns:
        int: -1 if version1 < version2, 0 if equal, 1 if version1 > version2
        
    Example:
        >>> result = compare_versions("0.1.0", "1.0.0")
        >>> assert result == -1  # 0.1.0 is less than 1.0.0
    """
    # Normalize to ConfigVersion objects
    if isinstance(version1, str):
        version1 = ConfigVersion.from_string(version1)
    if isinstance(version2, str):
        version2 = ConfigVersion.from_string(version2)
    
    sem_ver1 = version1.to_semantic_version()
    sem_ver2 = version2.to_semantic_version()
    
    if sem_ver1 < sem_ver2:
        return -1
    elif sem_ver1 > sem_ver2:
        return 1
    else:
        return 0


# Module initialization and validation
def _validate_compatibility_matrix() -> None:
    """Validate the compatibility matrix for consistency."""
    logger.debug("Validating compatibility matrix consistency")
    
    # Check that all enum versions are represented
    enum_versions = {v.value for v in ConfigVersion}
    matrix_versions = set(COMPATIBILITY_MATRIX.keys())
    
    if enum_versions != matrix_versions:
        logger.warning(f"Version mismatch: enum={enum_versions}, matrix={matrix_versions}")
    
    logger.debug("Compatibility matrix validation completed")


# Initialize module
_validate_compatibility_matrix()

logger.info(f"Version management module initialized - current version: {CURRENT_VERSION}")