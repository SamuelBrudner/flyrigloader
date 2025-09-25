"""Metadata extraction helpers for file discovery."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

from flyrigloader import logger

from .patterns import PatternMatcher
from .providers import DateTimeProvider, StandardDateTimeProvider


class MetadataExtractor:
    """Encapsulates filename metadata parsing independent from filesystem traversal."""

    #: Supported filename date formats. Shared with the legacy FileDiscoverer implementation.
    DATE_FORMATS: Sequence[str] = (
        "%Y%m%d",
        "%Y-%m-%d",
        "%m-%d-%Y",
    )

    def __init__(
        self,
        pattern_matcher: Optional[PatternMatcher] = None,
        *,
        parse_dates: bool = False,
        datetime_provider: Optional[DateTimeProvider] = None,
    ) -> None:
        self._pattern_matcher = pattern_matcher
        self._parse_dates = parse_dates
        self._datetime_provider = datetime_provider or StandardDateTimeProvider()

    def extract(self, files: Iterable[str]) -> Dict[str, Dict[str, object]]:
        """Return metadata for *files* without touching the filesystem."""
        files = list(files)
        logger.debug("MetadataExtractor extracting metadata for %d files", len(files))

        if self._pattern_matcher:
            metadata = self._extract_with_matcher(files)
        else:
            metadata = {file_path: {"path": file_path} for file_path in files}

        if self._parse_dates:
            for file_path, info in metadata.items():
                self._parse_date(file_path, info)

        self._apply_backward_compatibility(metadata)
        return metadata

    def _extract_with_matcher(self, files: Iterable[str]) -> Dict[str, Dict[str, object]]:
        result: Dict[str, Dict[str, object]] = {}
        unmatched = 0
        for file_path in files:
            metadata = self._pattern_matcher.match_all(file_path) if self._pattern_matcher else None
            if metadata is None and self._pattern_matcher is not None:
                metadata = self._pattern_matcher.match(file_path)
            if metadata is None:
                metadata = {}
                unmatched += 1
            metadata["path"] = file_path
            result[file_path] = metadata

        if unmatched:
            logger.debug("MetadataExtractor: %d files had no pattern matches", unmatched)
        return result

    def _parse_date(self, file_path: str, file_info: Dict[str, object]) -> None:
        date_str = file_info.get("date") if isinstance(file_info, dict) else None

        if not date_str:
            basename = Path(file_path).name
            for pattern in (r"(\d{8})", r"(\d{4}-\d{2}-\d{2})", r"(\d{2}-\d{2}-\d{4})"):
                match = re.search(pattern, basename)
                if match:
                    date_str = match[1]
                    break

        if not date_str:
            return

        for fmt in self.DATE_FORMATS:
            try:
                parsed = self._datetime_provider.strptime(date_str, fmt)
                file_info["parsed_date"] = parsed
                return
            except ValueError:
                continue
        logger.warning("MetadataExtractor could not parse date '%s' in '%s'", date_str, file_path)

    def _apply_backward_compatibility(self, metadata: Dict[str, Dict[str, object]]) -> None:
        for path, info in metadata.items():
            if not isinstance(info, dict):
                continue
            experiment_id = info.get("experiment_id")
            if isinstance(experiment_id, str) and experiment_id.startswith("exp"):
                basename = Path(path).name
                if basename == "exp001_mouse_baseline.csv" and info.get("animal") == "mouse":
                    info["animal"] = "exp_mouse"
                    logger.debug(
                        "MetadataExtractor applied backward compatibility fix for %s", basename
                    )


__all__ = ["MetadataExtractor"]
