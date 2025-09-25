"""Validators used by discovery-oriented configuration features."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from flyrigloader import logger

from .base import raise_validation_error


def _validator_name(name: str) -> str:
    return f"discovery.{name}"


def pattern_validation(pattern: Any) -> re.Pattern:
    """Validate and compile regex patterns with comprehensive error handling."""

    validator = _validator_name("pattern_validation")

    if isinstance(pattern, re.Pattern):
        logger.debug("Pattern already compiled: %s", pattern.pattern)
        return pattern

    if not isinstance(pattern, str):
        raise_validation_error(
            validator,
            f"Pattern must be string or compiled Pattern, got {type(pattern)}",
            input_type=str(type(pattern)),
        )

    if not pattern.strip():
        raise_validation_error(validator, "Pattern cannot be empty or whitespace-only")

    if len(pattern) > 1000:
        logger.warning("Excessive pattern length: %d characters", len(pattern))
        raise_validation_error(
            validator,
            "Pattern length exceeds maximum allowed limit",
            pattern_length=len(pattern),
        )

    dangerous_patterns = [
        r"\([^)]*\+[^)]*\+[^)]*\)",
        r"\([^)]*\*[^)]*\*[^)]*\)",
        r"\([^)]*\{[^}]*,[^}]*\}[^)]*\{[^}]*,[^}]*\}[^)]*\)",
    ]

    for dangerous in dangerous_patterns:
        if re.search(dangerous, pattern):
            logger.warning("Potentially dangerous pattern detected: %s", pattern)
            break

    try:
        compiled_pattern = re.compile(pattern)
        logger.debug("Pattern compiled successfully: %s", pattern)
        return compiled_pattern
    except re.error as exc:
        raise_validation_error(
            validator,
            f"Invalid regex pattern '{pattern}': {exc}",
            pattern=pattern,
            cause=exc,
        )
    except Exception as exc:  # pragma: no cover - defensive
        raise_validation_error(
            validator,
            f"Unexpected error compiling pattern '{pattern}': {exc}",
            pattern=pattern,
            cause=exc,
        )


def date_format_validator(date_input: Any, date_format: str = "%Y-%m-%d") -> bool:
    """Validate date format consistency for dates_vials structure."""

    validator = _validator_name("date_format_validator")

    if not isinstance(date_input, str):
        raise_validation_error(
            validator,
            f"Date input must be string, got {type(date_input)}",
            input_type=str(type(date_input)),
        )

    if not date_input.strip():
        raise_validation_error(validator, "Date cannot be empty or whitespace-only")

    if len(date_input) > 50:
        raise_validation_error(
            validator,
            "Date string exceeds maximum allowed length",
            path_length=len(date_input),
        )

    try:
        datetime.strptime(date_input, date_format)
        logger.debug(
            "Date parsed successfully with format %s: %s",
            date_format,
            date_input,
        )
        return True
    except ValueError as primary_error:
        logger.debug(
            "Primary format %s failed for %s: %s",
            date_format,
            date_input,
            primary_error,
        )

    alternative_formats = [
        "%Y-%m-%d",
        "%Y_%m_%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y%m%d",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
    ]

    for alt_format in alternative_formats:
        if alt_format == date_format:
            continue
        try:
            datetime.strptime(date_input, alt_format)
            logger.info(
                "Date parsed with alternative format %s: %s",
                alt_format,
                date_input,
            )
            return True
        except ValueError:
            continue

    raise_validation_error(
        validator,
        f"Date '{date_input}' does not match expected formats",
        date=date_input,
        expected=date_format,
    )
