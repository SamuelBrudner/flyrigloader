"""Tests for experiment manifest discovery using realistic configuration data."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from flyrigloader.discovery.files import discover_experiment_manifest


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
