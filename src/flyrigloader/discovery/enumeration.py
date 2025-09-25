"""Tools for enumerating files in discovery workflows."""
from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

from flyrigloader import logger

from .providers import FilesystemProvider, StandardFilesystemProvider

PathLikeIterable = Iterable[Union[str, os.PathLike, bytes]]


def _normalize_directory_argument(directory: Union[str, os.PathLike, PathLikeIterable]) -> List[str]:
    """Normalize directory inputs to a list of filesystem paths."""
    logger.debug("Normalizing directory argument: %s", directory)

    if isinstance(directory, (str, os.PathLike, bytes)):
        directories: Sequence[Union[str, os.PathLike, bytes]] = [directory]
    elif isinstance(directory, Iterable):
        directories = list(directory)
    else:
        raise TypeError("Directory must be a path-like object or an iterable of path-like objects.")

    normalized_directories: List[str] = []
    for entry in directories:
        try:
            normalized_directories.append(os.fspath(entry))
        except TypeError as exc:  # pragma: no cover - defensive logging
            logger.error("Invalid path-like entry encountered during normalization: %s", entry)
            raise TypeError("Directory iterable must contain only path-like objects.") from exc

    logger.debug("Normalized directories: %s", normalized_directories)
    return normalized_directories


class FileEnumerator:
    """Enumerate files from directories with optional filtering."""

    def __init__(
        self,
        filesystem_provider: Optional[FilesystemProvider] = None,
        *,
        test_mode: bool = False,
    ) -> None:
        self.filesystem_provider = filesystem_provider or StandardFilesystemProvider()
        self.test_mode = test_mode
        logger.debug(
            "Initialized FileEnumerator with provider=%s, test_mode=%s",
            type(self.filesystem_provider).__name__,
            test_mode,
        )

    def find_files(
        self,
        directory: Union[str, os.PathLike, PathLikeIterable],
        pattern: str,
        *,
        recursive: bool = False,
        extensions: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        mandatory_substrings: Optional[List[str]] = None,
    ) -> List[str]:
        """Find files matching the supplied constraints."""
        logger.debug("Starting file discovery with pattern='%s', recursive=%s", pattern, recursive)
        logger.debug("Search directories: %s", directory)
        logger.debug("Extensions filter: %s", extensions)
        logger.debug("Ignore patterns: %s", ignore_patterns)
        logger.debug("Mandatory substrings: %s", mandatory_substrings)

        directories = _normalize_directory_argument(directory)
        logger.debug("Processing %d directories after normalization", len(directories))

        all_matched_files: List[str] = []

        for dir_path in directories:
            directory_path = Path(dir_path)
            logger.debug("Searching in directory: %s", directory_path)

            if not self.filesystem_provider.exists(directory_path):
                logger.warning("Directory does not exist: %s", directory_path)
                continue

            try:
                matched_files = self._glob(directory_path, pattern, recursive)
                logger.debug("Found %d files in %s", len(matched_files), directory_path)
                all_matched_files.extend(str(file) for file in matched_files)
            except Exception as exc:
                logger.error("Error searching directory %s: %s", directory_path, exc)
                if not self.test_mode:
                    raise

        logger.debug("Total files found before filtering: %d", len(all_matched_files))
        filtered_files = self._apply_filters(
            all_matched_files,
            extensions=extensions,
            ignore_patterns=ignore_patterns,
            mandatory_substrings=mandatory_substrings,
        )
        logger.debug("Final file count after filtering: %d", len(filtered_files))
        return filtered_files

    def _glob(self, directory_path: Path, pattern: str, recursive: bool) -> List[Path]:
        if recursive and "**" not in pattern:
            clean_pattern = pattern.lstrip("./")
            logger.debug("Using recursive glob with pattern: %s", clean_pattern)
            return self.filesystem_provider.rglob(directory_path, clean_pattern)

        logger.debug("Using standard glob with pattern: %s", pattern)
        return self.filesystem_provider.glob(directory_path, pattern)

    def _apply_filters(
        self,
        files: List[str],
        *,
        extensions: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        mandatory_substrings: Optional[List[str]] = None,
    ) -> List[str]:
        logger.debug("Applying filters to %d files", len(files))
        filtered_files = files
        initial_count = len(files)

        if extensions is not None:
            logger.debug("Applying extension filter: %s", extensions)
            if not extensions:
                logger.debug("Empty extension list provided - returning no files")
                filtered_files = []
            else:
                ext_filters = [
                    (ext if ext.startswith(".") else f".{ext}").lower()
                    for ext in extensions
                ]
                logger.debug("Normalized extension filters: %s", ext_filters)
                filtered_files = [
                    path
                    for path in filtered_files
                    if any(path.lower().endswith(ext) for ext in ext_filters)
                ]
            logger.debug(
                "After extension filtering: %d files (removed %d)",
                len(filtered_files),
                initial_count - len(filtered_files),
            )
            initial_count = len(filtered_files)

        if ignore_patterns:
            logger.debug("Applying ignore patterns: %s", ignore_patterns)
            filtered_files = [
                path
                for path in filtered_files
                if all(not fnmatch.fnmatch(Path(path).name, pattern) for pattern in ignore_patterns)
            ]
            logger.debug(
                "After ignore pattern filtering: %d files (removed %d)",
                len(filtered_files),
                initial_count - len(filtered_files),
            )
            initial_count = len(filtered_files)

        if mandatory_substrings:
            logger.debug("Applying mandatory substring filter: %s", mandatory_substrings)
            filtered_files = [
                path
                for path in filtered_files
                if any(substring in path for substring in mandatory_substrings)
            ]
            logger.debug(
                "After mandatory substring filtering: %d files (removed %d)",
                len(filtered_files),
                initial_count - len(filtered_files),
            )

        logger.debug("Filtering complete: %d files remaining", len(filtered_files))
        return filtered_files


__all__ = ["FileEnumerator", "_normalize_directory_argument"]
