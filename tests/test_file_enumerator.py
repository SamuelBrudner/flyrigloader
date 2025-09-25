import os
from pathlib import Path

from flyrigloader.discovery.enumeration import FileEnumerator


def _make_file(root: Path, relative: str) -> Path:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("data", encoding="utf-8")
    return target


def test_enumerator_filters_extension_ignore_and_substrings(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _make_file(root, "mouse_a.csv")
    _make_file(root, "mouse_b.txt")
    _make_file(root, "temp_mouse.csv")
    _make_file(root, "nested/rat_a.csv")

    enumerator = FileEnumerator()

    results = enumerator.find_files(
        directory=root,
        pattern="*.csv",
        recursive=True,
        extensions=["csv"],
        ignore_patterns=["temp*"],
        mandatory_substrings=["mouse", os.path.join("nested", "rat")],
    )

    assert sorted(Path(path).name for path in results) == ["mouse_a.csv", "rat_a.csv"]


def test_enumerator_handles_missing_directories(tmp_path: Path) -> None:
    existing_dir = tmp_path / "existing"
    missing_dir = tmp_path / "missing"
    _make_file(existing_dir, "sample.csv")

    enumerator = FileEnumerator()

    results = enumerator.find_files(
        directory=[missing_dir, existing_dir],
        pattern="*.csv",
        recursive=False,
    )

    assert [Path(path).name for path in results] == ["sample.csv"]
