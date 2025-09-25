import logging
from pathlib import Path

import pytest

from flyrigloader.discovery.enumeration import FileEnumerator
from flyrigloader.discovery.files import discover_experiment_manifest, discover_files


def _write_file(path: Path, name: str) -> Path:
    target = path / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("content", encoding="utf-8")
    return target


@pytest.mark.parametrize("recursive", [False, True])
def test_file_enumerator_routine_logs_are_debug_only(tmp_path: Path, caplog, recursive: bool) -> None:
    data_dir = tmp_path / "data"
    _write_file(data_dir, "a.txt")
    _write_file(data_dir, "nested/b.txt")

    enumerator = FileEnumerator()

    caplog.set_level(logging.DEBUG)

    results = enumerator.find_files(
        directory=data_dir,
        pattern="*.txt",
        recursive=recursive,
    )

    expected = {str(data_dir / "a.txt")}
    if recursive:
        expected.add(str(data_dir / "nested" / "b.txt"))

    assert set(results) == expected

    info_messages = [record.message for record in caplog.records if record.levelno == logging.INFO]
    assert all("file discovery" not in message for message in info_messages), info_messages


def test_discover_files_routine_logs_are_debug_only(tmp_path: Path, caplog) -> None:
    data_dir = tmp_path / "data"
    _write_file(data_dir, "alpha.csv")
    _write_file(data_dir, "beta.csv")

    caplog.set_level(logging.DEBUG)

    results = discover_files(
        directory=str(data_dir),
        pattern="*.csv",
        recursive=False,
    )

    assert sorted(Path(result).name for result in results) == ["alpha.csv", "beta.csv"]

    info_messages = [record.message for record in caplog.records if record.levelno == logging.INFO]
    assert all("discover_files" not in message for message in info_messages), info_messages


def test_discover_manifest_routine_logs_are_debug_only(tmp_path: Path, caplog) -> None:
    base_dir = tmp_path / "major"
    dataset_dir = base_dir / "dataset_a"
    _write_file(dataset_dir, "result.csv")

    config = {
        "project": {
            "directories": {"major_data_directory": str(base_dir)},
            "file_extensions": ["csv"],
        },
        "datasets": {
            "dataset_a": {
                "patterns": ["*.csv"],
            }
        },
        "experiments": {
            "exp_a": {
                "datasets": ["dataset_a"],
            }
        },
    }

    caplog.set_level(logging.DEBUG)

    manifest = discover_experiment_manifest(
        config=config,
        experiment_name="exp_a",
        parse_dates=False,
        include_stats=False,
        enable_kedro_metadata=False,
        version_aware_patterns=False,
    )

    assert [Path(file_info.path).name for file_info in manifest.files] == ["result.csv"]

    info_messages = [record.message for record in caplog.records if record.levelno == logging.INFO]
    assert all("manifest" not in message.lower() for message in info_messages), info_messages
