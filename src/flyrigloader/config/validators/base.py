"""Core primitives for configuration validator modules."""
from __future__ import annotations

from typing import Any, Dict, Optional

from flyrigloader import logger
from flyrigloader.exceptions import ConfigError


class ConfigValidationError(ConfigError, ValueError):
    """Exception raised when a configuration validator rejects input."""

    def __init__(
        self,
        validator: str,
        message: str,
        *,
        error_code: str = "CONFIG_010",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        context_data: Dict[str, Any] = dict(context or {})
        context_data.setdefault("validator", validator)
        super().__init__(message, error_code=error_code, context=context_data)
        self.validator = validator

    def __str__(self) -> str:  # pragma: no cover - simple delegation
        base = super().__str__()
        return f"{base} (validator={self.validator})"


def log_validation_failure(
    validator: str,
    message: str,
    *,
    level: str = "error",
    **context: Any,
) -> None:
    """Emit a structured log message for a validation failure."""
    context_parts = ", ".join(f"{key}={value}" for key, value in context.items())
    log_message = (
        f"Validation failure in {validator}: {message} | "
        f"validator={validator}"
    )
    if context_parts:
        log_message = f"{log_message} | {context_parts}"

    log_method = getattr(logger, level.lower(), logger.error)
    log_method(log_message)


def raise_validation_error(
    validator: str,
    message: str,
    *,
    error_code: str = "CONFIG_010",
    cause: Optional[BaseException] = None,
    **context: Any,
) -> None:
    """Log and raise a ConfigValidationError with optional context."""
    if cause is not None:
        context.setdefault("cause", repr(cause))
    log_validation_failure(validator, message, **context)
    error = ConfigValidationError(
        validator=validator,
        message=message,
        error_code=error_code,
        context=context,
    )
    if cause is not None:
        raise error from cause
    raise error
