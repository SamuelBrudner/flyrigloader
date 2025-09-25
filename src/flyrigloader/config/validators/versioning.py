"""Validation helpers focused on configuration version management."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Union

from semantic_version import Version

from flyrigloader import logger

from ..versioning import CURRENT_SCHEMA_VERSION
from .base import ConfigValidationError, raise_validation_error


def _validator_name(name: str) -> str:
    return f"versioning.{name}"


def validate_version_format(version_string: Any) -> bool:
    """Validate that a version string conforms to semantic versioning standards."""

    validator = _validator_name("validate_version_format")

    if not isinstance(version_string, str):
        raise_validation_error(
            validator,
            f"Version string must be str, got {type(version_string)}",
            input_type=str(type(version_string)),
        )

    if not version_string.strip():
        raise_validation_error(
            validator,
            "Version string cannot be empty or whitespace-only",
        )

    if len(version_string) > 100:
        raise_validation_error(
            validator,
            "Version string exceeds maximum allowed length",
            length=len(version_string),
        )

    clean_version = version_string.strip()

    try:
        parsed_version = Version(clean_version)
    except Exception as exc:
        raise_validation_error(
            validator,
            f"Invalid semantic version format '{clean_version}': {exc}",
            version=clean_version,
            cause=exc,
        )

    if parsed_version.major < 0 or parsed_version.minor < 0 or parsed_version.patch < 0:
        raise_validation_error(
            validator,
            f"Version components must be non-negative integers: {clean_version}",
            version=clean_version,
        )

    logger.debug("Version format validation passed: %s", clean_version)
    return True


def validate_version_compatibility(
    config_version: str,
    system_version: str = CURRENT_SCHEMA_VERSION,
) -> Tuple[bool, str, Optional[List[str]]]:
    """Validate configuration version compatibility with the current system version."""

    validator = _validator_name("validate_version_compatibility")

    if not isinstance(config_version, str):
        raise_validation_error(
            validator,
            f"config_version must be string, got {type(config_version)}",
            input_type=str(type(config_version)),
        )
    if not isinstance(system_version, str):
        raise_validation_error(
            validator,
            f"system_version must be string, got {type(system_version)}",
            input_type=str(type(system_version)),
        )

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
            f"Configuration version {config_version} is newer than supported version {system_version}."
        )
        logger.error(message)
        return False, message, None

    if config_ver.major != system_ver.major:
        message = (
            f"Configuration major version {config_ver.major} is incompatible with system major version {system_ver.major}."
        )
        logger.error(message)
        return False, message, None

    if config_ver.minor < system_ver.minor:
        message = (
            f"Configuration minor version {config_ver.minor} is behind system minor version {system_ver.minor}."
        )
        logger.warning(message)
        return True, message, None

    message = (
        f"Configuration patch version {config_ver.patch} is behind system patch version {system_ver.patch}."
    )
    logger.info(message)
    return True, message, None


def validate_config_version(config_data: Union[Dict[str, Any], str]) -> Tuple[bool, str, str]:
    """Validate configuration data to ensure it targets a supported version."""

    validator = _validator_name("validate_config_version")

    try:
        version_str = _extract_version_from_config(config_data)
        validate_version_format(version_str)
    except Exception as exc:
        raise_validation_error(
            validator,
            "Failed to determine configuration version",
            cause=exc,
        )

    logger.info(f"Detected configuration version: {version_str}")

    compatibility, message, _ = validate_version_compatibility(version_str)

    if compatibility:
        logger.debug(f"Configuration version validation succeeded: {version_str}")
        return True, version_str, message

    return False, version_str, message


def _extract_version_from_config(config_data: Union[Dict[str, Any], str]) -> str:
    """Extract the schema version from configuration data."""

    if isinstance(config_data, dict):
        version = config_data.get("schema_version")
        if version is None:
            raise ValueError("Configuration missing 'schema_version' field")
        return str(version)

    if isinstance(config_data, str):
        try:
            data = json.loads(config_data)
        except json.JSONDecodeError as exc:
            raise ValueError("String configuration data must be valid JSON to extract version") from exc
        if not isinstance(data, dict):
            raise ValueError("Configuration data must deserialize to a dictionary")
        version = data.get("schema_version")
        if version is None:
            raise ValueError("Configuration missing 'schema_version' field")
        return str(version)

    raise TypeError(
        f"Configuration data must be a dictionary or JSON string, got {type(config_data)}"
    )


def validate_config_with_version(
    config_data: Union[Dict[str, Any], str],
    expected_version: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Comprehensively validate configuration data with version awareness."""

    validator = _validator_name("validate_config_with_version")

    if not isinstance(config_data, (dict, str)):
        raise_validation_error(
            validator,
            "config_data must be a dictionary or JSON string",
            input_type=str(type(config_data)),
        )

    validation_result: Dict[str, Any] = {
        "version": None,
        "compatible": False,
        "validation_messages": [],
        "warnings": [],
        "errors": [],
    }

    try:
        is_version_valid, detected_version, version_message = validate_config_version(
            config_data
        )
        validation_result["version"] = detected_version
        validation_result["validation_messages"].append(version_message)

        if expected_version is not None:
            validate_version_format(expected_version)
            if detected_version != expected_version:
                error_msg = (
                    f"Detected version {detected_version} does not match expected {expected_version}."
                )
                logger.error(error_msg)
                validation_result["errors"].append(error_msg)
                return False, validation_result

        is_compatible, compat_message, _ = validate_version_compatibility(
            detected_version
        )
        validation_result["compatible"] = is_compatible
        validation_result["validation_messages"].append(compat_message)

        if not is_compatible:
            logger.error(compat_message)
            validation_result["errors"].append(compat_message)
            return False, validation_result

        structure_valid, structure_messages = _validate_config_structure(
            config_data, detected_version
        )
        validation_result["validation_messages"].extend(structure_messages)

        if not structure_valid:
            validation_result["errors"].extend(structure_messages)
            return False, validation_result

        deprecation_warnings = _get_version_deprecation_warnings(detected_version)
        validation_result["warnings"].extend(deprecation_warnings)

        overall_valid = is_version_valid and structure_valid and is_compatible

        if overall_valid:
            logger.info(
                "Configuration validation passed for version %s", detected_version
            )
        else:
            logger.warning(
                "Configuration validation failed for version %s", detected_version
            )

        return overall_valid, validation_result

    except ConfigValidationError:
        raise
    except Exception as exc:
        raise_validation_error(
            validator,
            f"Configuration validation failed with error: {exc}",
            cause=exc,
        )


def _validate_config_structure(
    config_data: Union[Dict[str, Any], str],
    version: str,
) -> Tuple[bool, List[str]]:
    """Validate configuration structure based on version-specific requirements."""

    logger.debug("Validating configuration structure for version %s", version)

    messages: List[str] = []

    if isinstance(config_data, str):
        messages.append("String-based configuration passed basic structure validation")
        return True, messages

    if not isinstance(config_data, dict):
        messages.append("Configuration must be a dictionary structure")
        return False, messages

    version_obj = Version(version)

    if "project" not in config_data:
        messages.append("Configuration must contain 'project' section")
        return False, messages

    if version_obj >= Version("0.2.0") and "datasets" not in config_data:
        messages.append(f"Configuration version {version} requires 'datasets' section")

    if version_obj >= Version("1.0.0") and "schema_version" not in config_data:
        messages.append(
            f"Configuration version {version} requires top-level 'schema_version' field"
        )

    return True, messages


def _get_version_deprecation_warnings(version: str) -> List[str]:
    """Return warnings associated with deprecated configuration versions."""

    warnings: List[str] = []

    try:
        version_obj = Version(version)
    except Exception:
        return warnings

    if version_obj < Version("0.5.0"):
        warnings.append(
            "Configuration version is below 0.5.0 and may be deprecated in future releases"
        )

    if version_obj < Version("1.0.0"):
        warnings.append(
            "Configuration version is below 1.0.0; upgrade recommended for latest features"
        )

    return warnings
