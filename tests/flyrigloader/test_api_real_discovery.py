"""Integration tests for flyrigloader.api using real discovery logic."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


def _install_fail_fast_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure a fail-fast YAML stub is available when PyYAML isn't installed."""

    if "yaml" in sys.modules:
        return

    try:
        importlib.import_module("yaml")
    except ModuleNotFoundError:
        yaml_stub = types.ModuleType("yaml")

        def _unavailable(*_: object, **__: object) -> None:
            raise ModuleNotFoundError("PyYAML is required for this test.")

        yaml_stub.safe_load = _unavailable  # type: ignore[attr-defined]
        yaml_stub.safe_dump = _unavailable  # type: ignore[attr-defined]
        yaml_stub.YAMLError = RuntimeError  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "yaml", yaml_stub)


@pytest.fixture
def api_module(monkeypatch: pytest.MonkeyPatch):
    """Import ``flyrigloader.api`` with fail-fast optional dependency stubs."""

    _install_fail_fast_yaml(monkeypatch)

    # Ensure the src directory is importable when running without project-level config.
    src_root = Path(__file__).resolve().parents[2] / "src"
    monkeypatch.syspath_prepend(str(src_root))

    # Ensure we import the module fresh so our dependency stubs are used.
    for module_name in [
        "flyrigloader.config.yaml_config",
        "flyrigloader.api",
    ]:
        sys.modules.pop(module_name, None)

    module = importlib.import_module("flyrigloader.api")
    try:
        yield module
    finally:
        sys.modules.pop("flyrigloader.api", None)


def _example_config(base_directory: Path, pattern: str) -> dict:
    return {
        "project": {
            "directories": {"major_data_directory": str(base_directory)},
            "ignore_substrings": [],
            "mandatory_experiment_strings": [],
            "extraction_patterns": [pattern],
        },
        "experiments": {
            "real_exp": {
                "datasets": ["dataset_a"],
                "metadata": {"extraction_patterns": [pattern]},
                "filters": {
                    "ignore_substrings": [],
                    "mandatory_experiment_strings": [],
                },
            }
        },
        "datasets": {
            "dataset_a": {
                "patterns": ["*.csv"],
                "dates_vials": {"20240101": [1]},
                "metadata": {"extraction_patterns": [pattern]},
            }
        },
    }


def test_load_experiment_files_returns_metadata_bucket(api_module, tmp_path):
    pattern = r".*/(?P<experiment>\w+)_(?P<date>\d{8})_(?P<trial>\d+)\.csv"
    base_dir = tmp_path / "data"
    date_dir = base_dir / "20240101"
    date_dir.mkdir(parents=True)
    data_file = date_dir / "exp_20240101_1.csv"
    data_file.write_text("content")

    config = _example_config(base_dir, pattern)

    result = api_module.load_experiment_files(
        config=config,
        experiment_name="real_exp",
        pattern="*.csv",
        extract_metadata=True,
        parse_dates=True,
    )

    file_entry = result[str(data_file)]
    assert file_entry["metadata"]["experiment"] == "exp"
    assert file_entry["metadata"]["date"] == "20240101"
    assert file_entry["metadata"]["trial"] == "1"
    assert "parsed_date" in file_entry["metadata"]
    assert file_entry["path"] == str(data_file)


def test_load_dataset_files_returns_metadata_bucket(api_module, tmp_path):
    pattern = r".*/(?P<dataset>\w+)_(?P<date>\d{8})_(?P<trial>\d+)\.csv"
    base_dir = tmp_path / "data"
    date_dir = base_dir / "20240101"
    date_dir.mkdir(parents=True)
    data_file = date_dir / "dataset_20240101_1.csv"
    data_file.write_text("content")

    config = _example_config(base_dir, pattern)

    result = api_module.load_dataset_files(
        config=config,
        dataset_name="dataset_a",
        pattern="*.csv",
        extract_metadata=True,
        parse_dates=True,
    )

    file_entry = result[str(data_file)]
    assert file_entry["metadata"]["dataset"] == "dataset"
    assert file_entry["metadata"]["date"] == "20240101"
    assert file_entry["metadata"]["trial"] == "1"
    assert "parsed_date" in file_entry["metadata"]
    assert file_entry["path"] == str(data_file)
