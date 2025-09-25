"""Experiment-level discovery orchestrators."""

from __future__ import annotations

from collections.abc import Iterable as IterableABC, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from flyrigloader import logger
from flyrigloader.config.versioning import CURRENT_SCHEMA_VERSION
from flyrigloader.config.yaml_config import (
    get_dataset_info,
    get_experiment_info,
    get_extraction_patterns,
    get_ignore_patterns,
)
from flyrigloader.discovery.models import FileManifest
from flyrigloader.discovery.providers import (
    DateTimeProvider,
    FilesystemProvider,
)
from flyrigloader.discovery.patterns import PatternMatcher

from .discoverer import FileDiscoverer


def _convert_substring_to_glob(pattern: str) -> str:
    if not pattern:
        return pattern

    if any(char in pattern for char in ("*", "?")):
        return pattern

    if pattern == "._":
        return "*._*"

    if pattern.startswith('.'):
        return f"{pattern}*"

    return f"*{pattern}*"


def discover_experiment_manifest(
    config: Mapping[str, Any] | Any,
    experiment_name: str,
    patterns: Optional[List[str]] = None,
    parse_dates: bool = True,
    include_stats: bool = True,
    filesystem_provider: Optional[FilesystemProvider] = None,
    pattern_matcher: Optional[PatternMatcher] = None,
    datetime_provider: Optional[DateTimeProvider] = None,
    test_mode: bool = False,
    enable_kedro_metadata: bool = True,
    kedro_namespace: Optional[str] = None,
    kedro_tags: Optional[List[str]] = None,
    schema_version: Optional[str] = None,
    version_aware_patterns: bool = True,
) -> FileManifest:
    """Discover experiment files and return a metadata-only manifest."""

    logger.debug("Discovering experiment manifest for '%s'", experiment_name)

    if hasattr(config, "model_dump") and not isinstance(config, Mapping):
        config_mapping = config.model_dump()  # type: ignore[assignment]
    else:
        config_mapping = config  # type: ignore[assignment]

    if not isinstance(config_mapping, Mapping):
        raise TypeError(
            "Configuration for discover_experiment_manifest must be mapping-like or support model_dump()."
        )

    if "project" not in config_mapping:
        raise KeyError("Configuration missing 'project' section required for discovery")

    project_config = config_mapping["project"]
    if not isinstance(project_config, Mapping):
        raise TypeError("Project configuration must be mapping-like for discovery operations")

    directories_cfg = project_config.get("directories", {})
    if not isinstance(directories_cfg, Mapping):
        raise TypeError("Project directories configuration must be mapping-like")

    base_directory = directories_cfg.get("major_data_directory")
    if not base_directory:
        raise ValueError("Project configuration must define 'major_data_directory' for discovery")

    base_directory_path = Path(str(base_directory))

    experiment_info = get_experiment_info(config_mapping, experiment_name)
    dataset_names = experiment_info.get("datasets", [])
    if not dataset_names:
        raise ValueError(
            f"Experiment '{experiment_name}' does not reference any datasets; unable to perform discovery."
        )

    extraction_patterns = patterns or get_extraction_patterns(config_mapping, experiment_name)

    ignore_patterns = get_ignore_patterns(config_mapping, experiment_name)

    project_extensions = project_config.get("file_extensions")
    if project_extensions is None:
        extensions: Optional[List[str]] = None
    elif isinstance(project_extensions, IterableABC) and not isinstance(project_extensions, (str, bytes)):
        extensions = [str(ext) for ext in project_extensions]
    else:
        raise TypeError("project.file_extensions must be a list of extensions when provided")

    recursive = True
    dataset_targets: List[Dict[str, Any]] = []
    dataset_ignore_patterns: List[str] = []

    for dataset_name in dataset_names:
        dataset_info = get_dataset_info(config_mapping, dataset_name)

        raw_patterns = dataset_info.get("patterns") or ["*"]
        if isinstance(raw_patterns, IterableABC) and not isinstance(raw_patterns, (str, bytes)):
            dataset_patterns = [str(pattern) for pattern in raw_patterns]
        else:
            raise TypeError(
                f"Dataset '{dataset_name}' patterns must be provided as an iterable of strings"
            )

        dataset_filters = dataset_info.get("filters", {})
        if dataset_filters and not isinstance(dataset_filters, Mapping):
            raise TypeError(f"Dataset '{dataset_name}' filters must be mapping-like if provided")

        if isinstance(dataset_filters, Mapping):
            ignore_from_dataset = dataset_filters.get("ignore_substrings", [])
            if ignore_from_dataset:
                if not isinstance(ignore_from_dataset, IterableABC) or isinstance(
                    ignore_from_dataset, (str, bytes)
                ):
                    raise TypeError(
                        f"Dataset '{dataset_name}' ignore_substrings must be a list of strings"
                    )
                dataset_ignore_patterns.extend(
                    _convert_substring_to_glob(str(pattern)) for pattern in ignore_from_dataset
                )

        dataset_base_dir = base_directory_path / dataset_name
        directories: List[str] = []

        dates_vials = dataset_info.get("dates_vials")
        if isinstance(dates_vials, Mapping) and dates_vials:
            for date_key in dates_vials.keys():
                directories.append(str(dataset_base_dir / str(date_key)))
        else:
            directories.append(str(dataset_base_dir))

        dataset_targets.append(
            {
                "dataset": dataset_name,
                "directories": directories,
                "patterns": dataset_patterns,
            }
        )

    combined_ignore_patterns = list(dict.fromkeys(ignore_patterns + dataset_ignore_patterns))

    config_schema_version = schema_version
    if config_schema_version is None:
        config_schema_version = getattr(config, "schema_version", CURRENT_SCHEMA_VERSION)

    logger.debug("Using schema version: %s", config_schema_version)

    discoverer = FileDiscoverer(
        extract_patterns=extraction_patterns,
        parse_dates=parse_dates,
        include_stats=include_stats,
        filesystem_provider=filesystem_provider,
        pattern_matcher=pattern_matcher,
        datetime_provider=datetime_provider,
        test_mode=test_mode,
        enable_kedro_metadata=enable_kedro_metadata,
        kedro_namespace=kedro_namespace or experiment_name,
        kedro_tags=kedro_tags,
        schema_version=config_schema_version,
        version_aware_patterns=version_aware_patterns,
    )

    all_files = []
    seen_files: Set[str] = set()
    experiment_metadata = {
        "experiment_name": experiment_name,
        "base_directory": str(base_directory_path),
        "search_targets": dataset_targets,
        "discovery_settings": {
            "recursive": recursive,
            "parse_dates": parse_dates,
            "include_stats": include_stats,
            "extensions": extensions,
            "ignore_patterns": combined_ignore_patterns,
            "extraction_patterns": extraction_patterns or [],
        },
        "ignore_patterns": combined_ignore_patterns,
        "datasets": dataset_names,
    }

    for target in dataset_targets:
        dataset_name = target["dataset"]
        for directory in target["directories"]:
            for pattern in target["patterns"]:
                logger.debug(
                    "Discovering files for dataset '%s' in '%s' with pattern '%s'",
                    dataset_name,
                    directory,
                    pattern,
                )

                discovered = discoverer.discover(
                    directory=str(directory),
                    pattern=pattern,
                    recursive=recursive,
                    extensions=extensions,
                    ignore_patterns=combined_ignore_patterns,
                )

                if isinstance(discovered, dict):
                    for file_path, metadata in discovered.items():
                        if file_path in seen_files:
                            continue
                        seen_files.add(file_path)

                        file_info = discoverer.create_version_aware_file_info(file_path, metadata)
                        all_files.append(file_info)
                else:
                    for file_path in discovered:
                        if file_path in seen_files:
                            continue
                        seen_files.add(file_path)

                        file_info = discoverer.create_version_aware_file_info(file_path, {})
                        all_files.append(file_info)

    manifest = FileManifest(
        files=all_files,
        metadata=experiment_metadata,
        manifest_version=config_schema_version,
    )

    manifest.discovery_metadata = {
        "discovery_timestamp": datetime.now().isoformat(),
        "schema_version": config_schema_version,
        "kedro_enabled": enable_kedro_metadata,
        "version_aware_patterns": version_aware_patterns,
        "experiment_name": experiment_name,
        "datasets": dataset_names,
    }

    if enable_kedro_metadata:
        try:
            catalog_entries = manifest.generate_kedro_catalog_entries()
            logger.debug("Generated %d Kedro catalog entries", len(catalog_entries))

            is_compatible, issues = manifest.validate_kedro_compatibility()
            if not is_compatible:
                logger.warning("Kedro compatibility issues found: %s", issues)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Error generating Kedro catalog entries: %s", exc)

    logger.debug("Created enhanced manifest with %d files for experiment '%s'", len(all_files), experiment_name)
    version_summary = manifest.get_version_summary()
    logger.debug("Manifest version summary: %s", version_summary)

    return manifest
