"""Provider interfaces and implementations for discovery components."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, List, Protocol

from flyrigloader import logger


class FilesystemProvider(Protocol):
    """Protocol for filesystem operations to enable dependency injection."""

    def glob(self, path: Path, pattern: str) -> List[Path]:
        """Execute glob operation on the given path."""

    def rglob(self, path: Path, pattern: str) -> List[Path]:
        """Execute recursive glob operation on the given path."""

    def stat(self, path: Path) -> Any:
        """Get file statistics for the given path."""

    def exists(self, path: Path) -> bool:
        """Check if path exists."""


class StandardFilesystemProvider:
    """Standard filesystem provider using pathlib operations."""

    def glob(self, path: Path, pattern: str) -> List[Path]:
        """Execute glob operation using pathlib."""
        try:
            logger.debug("Performing glob search: %s with pattern: %s", path, pattern)
            result = list(path.glob(pattern))
            logger.debug("Glob search found %d files", len(result))
            return result
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Error during glob operation: %s", exc)
            raise

    def rglob(self, path: Path, pattern: str) -> List[Path]:
        """Execute recursive glob operation using pathlib."""
        try:
            logger.debug("Performing recursive glob search: %s with pattern: %s", path, pattern)
            result = list(path.rglob(pattern))
            logger.debug("Recursive glob search found %d files", len(result))
            return result
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Error during recursive glob operation: %s", exc)
            raise

    def stat(self, path: Path) -> Any:
        """Get file statistics using pathlib."""
        try:
            return path.stat()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Error getting file stats for %s: %s", path, exc)
            raise

    def exists(self, path: Path) -> bool:
        """Check if path exists using pathlib."""
        try:
            return path.exists()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Error checking path existence for %s: %s", path, exc)
            raise


class DateTimeProvider(Protocol):
    """Protocol for datetime operations to enable dependency injection."""

    def strptime(self, date_string: str, format_string: str) -> datetime:
        """Parse date string using the specified format."""

    def now(self) -> datetime:
        """Get current datetime."""


class StandardDateTimeProvider:
    """Standard datetime provider using the :mod:`datetime` module."""

    def strptime(self, date_string: str, format_string: str) -> datetime:
        """Parse date string using :func:`datetime.datetime.strptime`."""
        try:
            return datetime.strptime(date_string, format_string)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Error parsing date string '%s' with format '%s': %s", date_string, format_string, exc)
            raise

    def now(self) -> datetime:
        """Return the current datetime."""
        return datetime.now()


__all__ = [
    "FilesystemProvider",
    "StandardFilesystemProvider",
    "DateTimeProvider",
    "StandardDateTimeProvider",
]
