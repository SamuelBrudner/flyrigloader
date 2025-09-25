"""Manifest discovery helpers for :mod:`flyrigloader.api`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from semantic_version import Version

from flyrigloader import logger
from flyrigloader.config.models import LegacyConfigAdapter
from flyrigloader.config.validators import validate_config_version
from flyrigloader.config.versioning import CURRENT_SCHEMA_VERSION
from flyrigloader.discovery.files import discover_experiment_manifest as _discover_experiment_manifest
from flyrigloader.exceptions import FlyRigLoaderError

from ._shared import resolve_api_override
from .config import (
    _coerce_config_for_version_validation,
    _load_and_validate_config,
    _resolve_config_source,
)
from .dependencies import DefaultDependencyProvider, get_dependency_provider


def discover_experiment_manifest(
    config: Optional[Union[Dict[str, Any], LegacyConfigAdapter, Any]] = None,
    experiment_name: str = "",
    config_path: Optional[Union[str, Path]] = None,
    base_directory: Optional[Union[str, Path]] = None,
    pattern: str = "*.*",
    recursive: bool = True,
    extensions: Optional[List[str]] = None,
    extract_metadata: bool = True,
    parse_dates: bool = True,
    _deps: Optional[DefaultDependencyProvider] = None,
) -> Dict[str, Dict[str, Any]]:
    """Discover experiment files and return a comprehensive manifest."""

    operation_name = "discover_experiment_manifest"

    if _deps is None:
        _deps = get_dependency_provider()

    logger.info(f"🔍 Discovering experiment manifest for '{experiment_name}'")
    logger.debug(
        "Discovery parameters: base_directory=%s, pattern=%s, recursive=%s, extensions=%s, "
        "extract_metadata=%s, parse_dates=%s",
        base_directory,
        pattern,
        recursive,
        extensions,
        extract_metadata,
        parse_dates,
    )

    if not experiment_name or not isinstance(experiment_name, str):
        error_msg = (
            f"Invalid experiment_name for {operation_name}: '{experiment_name}'. "
            "experiment_name must be a non-empty string representing the experiment identifier."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)

    config_dict = _resolve_config_source(config, config_path, operation_name, _deps)

    try:
        logger.debug("Starting decoupled file discovery for experiment '%s'", experiment_name)

        discover_manifest = resolve_api_override(
            globals(), "_discover_experiment_manifest", _discover_experiment_manifest
        )

        file_manifest = discover_manifest(
            config=config_dict,
            experiment_name=experiment_name,
            patterns=None,
            parse_dates=parse_dates,
            include_stats=extract_metadata,
            test_mode=False,
        )

        manifest_dict = file_manifest.to_legacy_dict()

        file_count = len(manifest_dict)
        total_size = sum(item.get("size", 0) for item in manifest_dict.values())
        logger.info("✓ Discovered %s files for experiment '%s'", file_count, experiment_name)
        logger.info("  Total data size: %s bytes (%0.1f MB)", total_size, total_size / (1024**2) if total_size else 0.0)
        sample_files = list(manifest_dict.keys())[:3]
        suffix = "..." if file_count > 3 else ""
        logger.debug("  Sample files: %s%s", sample_files, suffix)

        return manifest_dict

    except Exception as exc:
        error_msg = (
            f"Failed to discover experiment manifest for '{experiment_name}': {exc}. "
            "Please check the experiment configuration and data directory structure."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg) from exc


def validate_manifest(
    manifest: Dict[str, Dict[str, Any]],
    config: Optional[Union[Dict[str, Any], Any]] = None,
    config_path: Optional[Union[str, Path]] = None,
    strict_validation: bool = False,
    _deps: Optional[DefaultDependencyProvider] = None,
) -> Dict[str, Any]:
    """Validate an experiment manifest for pre-flight validation without side effects."""

    operation_name = "validate_manifest"

    if _deps is None:
        _deps = get_dependency_provider()

    logger.info(f"🔍 Validating experiment manifest with {len(manifest)} files")
    logger.debug("Validation parameters: strict_validation=%s", strict_validation)

    if not isinstance(manifest, dict):
        error_msg = (
            f"Invalid manifest parameter for {operation_name}: expected dict, got {type(manifest).__name__}. "
            "manifest must be a dictionary from discover_experiment_manifest()."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)

    if not manifest:
        logger.warning("Empty manifest provided for validation")
        return {
            "valid": True,
            "file_count": 0,
            "errors": [],
            "warnings": ["Empty manifest - no files to validate"],
            "metadata": {"validation_type": "empty_manifest"},
        }

    validation_report: Dict[str, Any] = {
        "valid": True,
        "file_count": len(manifest),
        "errors": [],
        "warnings": [],
        "metadata": {
            "validation_type": "strict" if strict_validation else "basic",
            "validated_files": [],
            "failed_files": [],
            "total_size_bytes": 0,
        },
    }

    try:
        config_dict = None
        if config is not None or config_path is not None:
            try:
                if config is not None:
                    config_dict = config
                    logger.debug("Using provided configuration for validation rules")
                else:
                    config_dict = _load_and_validate_config(config_path, None, operation_name, _deps)
                    logger.debug(f"Loaded configuration from {config_path} for validation")
            except Exception as exc:
                validation_report["warnings"].append(
                    f"Failed to load configuration for validation: {exc}"
                )
                logger.warning("Configuration loading failed, proceeding with basic validation: %s", exc)

        for file_path, file_metadata in manifest.items():
            try:
                logger.debug("Validating file: %s", file_path)

                if not isinstance(file_metadata, dict):
                    error_msg = (
                        f"Invalid metadata for file {file_path}: expected dict, got {type(file_metadata).__name__}"
                    )
                    validation_report["errors"].append(error_msg)
                    validation_report["metadata"]["failed_files"].append(file_path)
                    continue

                required_fields = ["path", "size"]
                for field in required_fields:
                    if field not in file_metadata:
                        validation_report["warnings"].append(
                            f"Missing metadata field '{field}' for file {file_path}"
                        )

                if strict_validation:
                    file_path_obj = Path(file_path)
                    if not file_path_obj.exists():
                        error_msg = f"File does not exist: {file_path}"
                        validation_report["errors"].append(error_msg)
                        validation_report["metadata"]["failed_files"].append(file_path)
                        continue

                    if not file_path_obj.is_file():
                        error_msg = f"Path is not a regular file: {file_path}"
                        validation_report["errors"].append(error_msg)
                        validation_report["metadata"]["failed_files"].append(file_path)
                        continue

                    actual_size = file_path_obj.stat().st_size
                    reported_size = file_metadata.get("size", 0)
                    if abs(actual_size - reported_size) > 1024:
                        validation_report["warnings"].append(
                            f"Size mismatch for {file_path}: reported {reported_size}, actual {actual_size}"
                        )

                file_size = file_metadata.get("size", 0)
                if isinstance(file_size, (int, float)):
                    validation_report["metadata"]["total_size_bytes"] += file_size

                validation_report["metadata"]["validated_files"].append(file_path)

            except Exception as exc:
                error_msg = f"Validation failed for file {file_path}: {exc}"
                validation_report["errors"].append(error_msg)
                validation_report["metadata"]["failed_files"].append(file_path)
                logger.error(error_msg)

        if config_dict is not None:
            try:
                normalized_config = _coerce_config_for_version_validation(config_dict)
                is_valid, detected_version, message = validate_config_version(normalized_config)
                validation_report["metadata"]["config_version"] = str(detected_version)
                logger.debug("Detected configuration version: %s", detected_version)

                current_version = Version(CURRENT_SCHEMA_VERSION)
                config_version = Version(str(detected_version))

                if not is_valid:
                    validation_report["errors"].append(message)
                elif config_version < current_version:
                    validation_report["warnings"].append(
                        f"Configuration version {detected_version} is older than supported version {CURRENT_SCHEMA_VERSION}."
                    )
                elif config_version > current_version:
                    validation_report["errors"].append(
                        f"Configuration version {detected_version} is newer than supported version {CURRENT_SCHEMA_VERSION}. "
                        "Please upgrade FlyRigLoader."
                    )

            except Exception as exc:
                validation_report["warnings"].append(
                    f"Configuration version validation failed: {exc}"
                )
                logger.warning("Configuration version validation error: %s", exc)

        validation_report["valid"] = len(validation_report["errors"]) == 0

        if validation_report["valid"]:
            logger.info(
                "✓ Manifest validation successful: %s files validated", validation_report["file_count"]
            )
            if validation_report["warnings"]:
                logger.info("  Warnings: %s", len(validation_report["warnings"]))
        else:
            logger.error(
                "✗ Manifest validation failed: %s errors, %s warnings",
                len(validation_report["errors"]),
                len(validation_report["warnings"]),
            )

        logger.debug(
            "Validation summary: %s bytes total",
            validation_report["metadata"]["total_size_bytes"],
        )

        return validation_report

    except Exception as exc:
        error_msg = (
            f"Manifest validation failed for {operation_name}: {exc}. "
            "Please check the manifest structure and validation parameters."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from exc


__all__ = ["discover_experiment_manifest", "validate_manifest"]

