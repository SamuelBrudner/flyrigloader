"""Schema version constants and helpers without automatic upgrade support."""

from typing import Final

from semantic_version import Version

CURRENT_SCHEMA_VERSION: Final[str] = "1.0.0"


def parse_schema_version(version: str) -> Version:
    """Parse a schema version string into a :class:`semantic_version.Version`."""
    if not isinstance(version, str):
        raise TypeError(f"schema version must be a string, got {type(version)}")
    try:
        return Version(version)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Invalid schema version '{version}': {exc}") from exc


def is_supported_version(version: str) -> bool:
    """Return ``True`` when *version* matches :data:`CURRENT_SCHEMA_VERSION`."""
    return parse_schema_version(version) == Version(CURRENT_SCHEMA_VERSION)
