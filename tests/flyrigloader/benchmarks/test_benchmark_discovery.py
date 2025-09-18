"""Behavioral tests for file discovery utilities."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List

import pytest

from flyrigloader.discovery.files import discover_files, get_latest_file


@pytest.fixture(name="sample_tree")
def fixture_sample_tree(tmp_path: Path) -> List[Path]:
    """Create a small directory tree for discovery tests."""
    (tmp_path / "nested").mkdir()
    files = [
        tmp_path / "data1.csv",
        tmp_path / "data2.txt",
        tmp_path / "nested" / "data3.csv",
    ]
    for index, file_path in enumerate(files, start=1):
        file_path.write_text(f"payload {index}\n", encoding="utf-8")
    return files


def test_discover_files_basic_listing(sample_tree: List[Path]) -> None:
    """discover_files should return matching paths for a glob pattern."""
    base_dir = sample_tree[0].parent
    discovered = discover_files(str(base_dir), pattern="*.csv", recursive=True)
    assert sorted(Path(path) for path in discovered) == sorted(
        path for path in sample_tree if path.suffix == ".csv"
    )


def test_discover_files_extension_filter(sample_tree: List[Path]) -> None:
    """Explicit extension filters should restrict the results."""
    base_dir = sample_tree[0].parent
    discovered = discover_files(
        str(base_dir),
        pattern="*",
        recursive=True,
        extensions=[".txt"],
    )
    assert sorted(Path(path) for path in discovered) == sorted(
        path for path in sample_tree if path.suffix == ".txt"
    )


def test_get_latest_file_returns_none_for_empty_list() -> None:
    """get_latest_file should handle empty inputs explicitly."""
    assert get_latest_file([]) is None


def test_get_latest_file_identifies_most_recent(sample_tree: List[Path]) -> None:
    """The most recently updated file should be returned."""
    base_dir = sample_tree[0].parent
    newest = base_dir / "newest.csv"
    newest.write_text("fresh\n", encoding="utf-8")
    future_timestamp = time.time() + 10
    os.utime(newest, (future_timestamp, future_timestamp))
    discovered = [str(path) for path in sample_tree] + [str(newest)]

    assert get_latest_file(discovered) == str(newest)
