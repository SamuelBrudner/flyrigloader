"""Data models representing discovery results."""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from semantic_version import Version

from flyrigloader.config.versioning import CURRENT_SCHEMA_VERSION


@dataclass
class FileInfo:
    """Metadata about a discovered file without loading its contents."""

    path: str
    size: Optional[int] = None
    mtime: Optional[float] = None
    ctime: Optional[float] = None
    creation_time: Optional[float] = None
    extracted_metadata: Dict[str, Any] = field(default_factory=dict)
    parsed_date: Optional[datetime] = None

    kedro_dataset_name: Optional[str] = None
    kedro_version: Optional[str] = None
    kedro_namespace: Optional[str] = None
    kedro_tags: List[str] = field(default_factory=list)
    catalog_metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = CURRENT_SCHEMA_VERSION
    version_compatibility: Optional[Dict[str, bool]] = field(default_factory=dict)

    def get_kedro_dataset_path(self) -> Optional[str]:
        if self.kedro_dataset_name:
            if self.kedro_namespace:
                return f"{self.kedro_namespace}.{self.kedro_dataset_name}"
            return self.kedro_dataset_name
        return None

    def is_kedro_versioned(self) -> bool:
        return bool(self.kedro_version and self.kedro_version != "latest")

    def get_version_info(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kedro_version": self.kedro_version,
            "version_compatibility": self.version_compatibility or {},
            "is_versioned": self.is_kedro_versioned(),
        }


@dataclass
class FileStatistics:
    total_files: int
    total_size: int
    file_types: Dict[str, int] = field(default_factory=dict)
    date_range: Optional[Tuple[datetime, datetime]] = None
    discovery_time: Optional[float] = None


@dataclass
class FileManifest:
    """Container aggregating discovered files and derived metadata."""

    files: List[FileInfo]
    metadata: Dict[str, Any] = field(default_factory=dict)
    statistics: Optional[FileStatistics] = None
    kedro_catalog_entries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    kedro_pipeline_compatibility: bool = True
    supported_kedro_versions: List[str] = field(default_factory=list)
    manifest_version: str = CURRENT_SCHEMA_VERSION
    discovery_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.statistics is None and self.files:
            self._compute_statistics()

    def _compute_statistics(self) -> None:
        if not self.files:
            return

        total_files = len(self.files)
        total_size = sum(f.size or 0 for f in self.files)
        file_types: Dict[str, int] = {}
        dates: List[datetime] = []

        for file_info in self.files:
            path = Path(file_info.path)
            ext = path.suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + 1

            if file_info.parsed_date:
                dates.append(file_info.parsed_date)

        date_range: Optional[Tuple[datetime, datetime]] = None
        if dates:
            dates.sort()
            date_range = (dates[0], dates[-1])

        self.statistics = FileStatistics(
            total_files=total_files,
            total_size=total_size,
            file_types=file_types,
            date_range=date_range,
        )

    def get_files_by_type(self, extension: str) -> List[FileInfo]:
        return [f for f in self.files if f.path.lower().endswith(extension.lower())]

    def get_files_by_pattern(self, pattern: str) -> List[FileInfo]:
        return [f for f in self.files if fnmatch.fnmatch(Path(f.path).name, pattern)]

    def get_kedro_compatible_files(self) -> List[FileInfo]:
        return [f for f in self.files if f.kedro_dataset_name or self._is_kedro_pattern(f.path)]

    def get_versioned_files(self) -> List[FileInfo]:
        return [f for f in self.files if f.is_kedro_versioned()]

    def get_files_by_namespace(self, namespace: str) -> List[FileInfo]:
        return [f for f in self.files if f.kedro_namespace == namespace]

    def generate_kedro_catalog_entries(self) -> Dict[str, Dict[str, Any]]:
        catalog_entries: Dict[str, Dict[str, Any]] = {}

        for file_info in self.files:
            if file_info.kedro_dataset_name:
                entry_name = file_info.get_kedro_dataset_path() or file_info.kedro_dataset_name

                catalog_entries[entry_name] = {
                    "type": "flyrigloader.kedro.datasets.FlyRigLoaderDataSet",
                    "filepath": file_info.path,
                    "metadata": file_info.catalog_metadata.copy(),
                    "versioned": file_info.is_kedro_versioned(),
                    "tags": file_info.kedro_tags.copy(),
                }

                if file_info.kedro_version:
                    catalog_entries[entry_name]["version"] = file_info.kedro_version

        self.kedro_catalog_entries = catalog_entries
        return catalog_entries

    def validate_kedro_compatibility(self, kedro_version: str = "0.18.0") -> Tuple[bool, List[str]]:
        issues: List[str] = []

        try:
            target_version = Version(kedro_version)
            min_supported = Version("0.18.0")

            if target_version < min_supported:
                issues.append(
                    f"Kedro version {kedro_version} is below minimum supported version {min_supported}"
                )
        except Exception:
            issues.append(f"Invalid Kedro version format: {kedro_version}")

        kedro_files = self.get_kedro_compatible_files()
        if not kedro_files and self.files:
            issues.append("No Kedro-compatible files found in manifest")

        for file_info in kedro_files:
            if file_info.kedro_dataset_name and not self._is_valid_kedro_name(file_info.kedro_dataset_name):
                issues.append(f"Invalid Kedro dataset name: {file_info.kedro_dataset_name}")

        self.kedro_pipeline_compatibility = len(issues) == 0
        return self.kedro_pipeline_compatibility, issues

    def get_version_summary(self) -> Dict[str, Any]:
        version_counts: Dict[str, int] = {}
        for file_info in self.files:
            version = file_info.schema_version
            version_counts[version] = version_counts.get(version, 0) + 1

        return {
            "manifest_version": self.manifest_version,
            "file_version_distribution": version_counts,
            "kedro_compatible_count": len(self.get_kedro_compatible_files()),
            "versioned_files_count": len(self.get_versioned_files()),
            "discovery_metadata": self.discovery_metadata,
        }

    def _is_kedro_pattern(self, filepath: str) -> bool:
        path = Path(filepath)

        kedro_version_pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}\.\d{2}\.\d{2}\.\d{3}Z$"
        if re.search(kedro_version_pattern, path.parent.name):
            return True

        kedro_name_pattern = r"^[a-zA-Z][a-zA-Z0-9_]*(\.[a-zA-Z][a-zA-Z0-9_]*)*$"
        if "." in path.stem and re.match(kedro_name_pattern, path.stem):
            return True

        return False

    def _is_valid_kedro_name(self, name: str) -> bool:
        pattern = r"^[a-zA-Z][a-zA-Z0-9_]*(\.[a-zA-Z][a-zA-Z0-9_]*)*$"
        return bool(re.match(pattern, name))


__all__ = ["FileInfo", "FileStatistics", "FileManifest"]
