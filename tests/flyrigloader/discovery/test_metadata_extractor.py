from datetime import datetime
from pathlib import Path

import pytest

class StubDateTimeProvider:
    """Simple date parser used in tests to make expectations deterministic."""

    def strptime(self, date_string: str, format_string: str) -> datetime:
        return datetime.strptime(date_string, format_string)

    def now(self) -> datetime:
        return datetime.now()


@pytest.fixture
def sample_files(tmp_path: Path) -> list[str]:
    files = [
        tmp_path / "mouse_20240101_control_1.csv",
        tmp_path / "mouse_report.txt",
    ]
    for path in files:
        path.write_text("test")
    return [str(path) for path in files]


def test_metadata_extractor_parses_dates(sample_files):
    """Metadata extraction should parse named groups and resolve dates independently of discovery."""
    class StubMatcher:
        def match_all(self, path: str):
            if path.endswith("mouse_20240101_control_1.csv"):
                return {
                    "animal": "mouse",
                    "date": "20240101",
                    "condition": "control",
                    "replicate": "1",
                }
            return None

        def match(self, path: str):
            return self.match_all(path)

    from flyrigloader.discovery.metadata import MetadataExtractor

    extractor = MetadataExtractor(
        pattern_matcher=StubMatcher(),
        parse_dates=True,
        datetime_provider=StubDateTimeProvider(),
    )

    metadata = extractor.extract(sample_files)

    matched = metadata[sample_files[0]]
    assert matched["animal"] == "mouse"
    assert matched["condition"] == "control"
    assert matched["replicate"] == "1"
    assert matched["parsed_date"].date() == datetime(2024, 1, 1).date()


def test_metadata_extractor_returns_path_when_no_match(sample_files):
    """Unmatched files should still produce a metadata envelope with only the path."""
    class StubMatcher:
        def match_all(self, path: str):
            return None

        def match(self, path: str):
            return None

    from flyrigloader.discovery.metadata import MetadataExtractor

    extractor = MetadataExtractor(pattern_matcher=StubMatcher(), parse_dates=True, datetime_provider=StubDateTimeProvider())

    metadata = extractor.extract(sample_files)
    assert metadata[sample_files[1]] == {"path": sample_files[1]}
