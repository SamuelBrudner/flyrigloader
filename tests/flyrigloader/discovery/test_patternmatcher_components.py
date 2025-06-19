"""Tests for PatternMatcher path component targeting."""
from pathlib import Path
import pytest

from flyrigloader.discovery.patterns import PatternMatcher


@pytest.mark.parametrize(
    "pattern,target_file,expected",
    [
        (
            "filename::^(?P<stem>.+)\\.txt$",
            Path("/tmp/example/file123.txt"),
            {"stem": "file123"},
        ),
        (
            "parent::^(?P<dir>.+)$",
            Path("/tmp/parent_dir/file.dat"),
            {"dir": "parent_dir"},
        ),
        (
            "parts[1]::^(?P<root>tmp)$",
            Path("/tmp/parent/child/file.dat"),
            {"root": "tmp"},
        ),
        (
            "parts[-2]::^(?P<child>child)$",
            Path("/tmp/parent/child/file.dat"),
            {"child": "child"},
        ),
    ],
)
def test_patternmatcher_component_matching(pattern, target_file, expected):
    matcher = PatternMatcher([pattern])
    result = matcher.match(str(target_file))
    assert result == expected


def test_out_of_range_parts_index(tmp_path: Path):
    file_path = tmp_path / "a" / "b" / "c.txt"
    file_path.parent.mkdir(parents=True)
    file_path.touch()

    # index 10 is out of range – should safely return None
    matcher = PatternMatcher(["parts[10]::^irrelevant$"])
    assert matcher.match(str(file_path)) is None


def test_backward_compat_full_path(tmp_path: Path):
    file_path = tmp_path / "animal_mouse_001.dat"
    file_path.touch()
    pattern = r"animal_mouse_(?P<id>\d+)\.dat$"
    matcher = PatternMatcher([pattern])
    result = matcher.match(str(file_path))
    assert result == {"id": "001"}
