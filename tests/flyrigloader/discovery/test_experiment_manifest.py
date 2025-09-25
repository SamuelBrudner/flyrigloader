"""Tests for experiment manifest discovery using realistic configuration data."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pytest

from flyrigloader.discovery.files import (
    FileInfo,
    FileManifest,
    discover_experiment_manifest,
)


@pytest.mark.usefixtures("comprehensive_sample_config_dict")
def test_discover_experiment_manifest_uses_dataset_definitions(
    tmp_path: Path,
    comprehensive_sample_config_dict: dict,
):
    """Ensure experiment manifest discovery respects dataset directories and patterns."""

    config = deepcopy(comprehensive_sample_config_dict)

    major_data_directory = tmp_path / "neuro_data"
    config["project"]["directories"]["major_data_directory"] = str(major_data_directory)

    baseline_date_dir = major_data_directory / "baseline_behavior" / "2024-12-20"
    baseline_date_dir.mkdir(parents=True)

    matching_file = baseline_date_dir / "baseline_behavior_20241220.csv"
    matching_file.write_text("data")

    ignored_file = baseline_date_dir / "static_horiz_ribbon_baseline.csv"
    ignored_file.write_text("data")

    opto_dir = major_data_directory / "optogenetic_stimulation" / "2024-12-18"
    opto_dir.mkdir(parents=True)
    opto_file = opto_dir / "opto_trial_20241218.csv"
    opto_file.write_text("data")

    manifest = discover_experiment_manifest(
        config=config,
        experiment_name="baseline_control_study",
        parse_dates=False,
        include_stats=False,
    )

    discovered_paths = {Path(info.path) for info in manifest.files}

    assert matching_file in discovered_paths
    assert ignored_file not in discovered_paths
    assert opto_file not in discovered_paths

    dataset_targets = {target["dataset"]: target for target in manifest.metadata["search_targets"]}
    baseline_target = dataset_targets["baseline_behavior"]

    assert str(baseline_date_dir) in baseline_target["directories"]
    assert "*baseline*" in baseline_target["patterns"]
    assert any("static_horiz_ribbon" in pattern for pattern in manifest.metadata["ignore_patterns"])


def test_file_manifest_to_legacy_dict_includes_expected_fields() -> None:
    """``FileManifest.to_legacy_dict`` should capture metadata expected by API layer."""

    info = FileInfo(
        path="/data/sample.pkl",
        size=1024,
        mtime=123.4,
        ctime=125.6,
        creation_time=111.1,
        extracted_metadata={"trial": 5, "subject": "alpha"},
        parsed_date=datetime(2024, 1, 2, 3, 4, 5),
        kedro_dataset_name="dataset",
        kedro_namespace="namespace",
        kedro_tags=["tag1", "tag2"],
        catalog_metadata={"compression": "lz4"},
        kedro_version="2024.1",
        version_compatibility={"kedro>=0.18": True},
    )

    manifest = FileManifest(files=[info])

    legacy = manifest.to_legacy_dict()
    assert list(legacy.keys()) == [info.path]

    entry = legacy[info.path]
    assert entry["path"] == info.path
    assert entry["size"] == info.size
    assert entry["metadata"] == {"trial": 5, "subject": "alpha"}
    assert entry["parsed_dates"] == {"parsed_date": info.parsed_date}
    assert entry["mtime"] == info.mtime
    assert entry["ctime"] == info.ctime
    assert entry["creation_time"] == info.creation_time
    assert entry["kedro_dataset_name"] == info.kedro_dataset_name
    assert entry["kedro_namespace"] == info.kedro_namespace
    assert entry["kedro_tags"] == info.kedro_tags
    assert entry["catalog_metadata"] == info.catalog_metadata
    assert entry["kedro_version"] == info.kedro_version
    assert entry["version_compatibility"] == info.version_compatibility
    assert entry["schema_version"] == info.schema_version

    # Ensure nested structures are copies so callers can't mutate internals inadvertently.
    entry["metadata"]["trial"] = 7
    assert info.extracted_metadata["trial"] == 5
