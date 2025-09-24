"""Optional Kedro integration helpers for :mod:`flyrigloader.api`."""

from __future__ import annotations

from functools import lru_cache
from importlib import util
from typing import Any, Optional, Type

from flyrigloader import logger
from flyrigloader.exceptions import FlyRigLoaderError

_KEDRO_IMPORT_ERROR: Optional[ModuleNotFoundError] = None


def _find_kedro_spec():
    """Return the import spec for Kedro if it can be discovered."""

    try:
        return util.find_spec("kedro")
    except (ModuleNotFoundError, ValueError):
        return None


@lru_cache(maxsize=1)
def _import_dataset_class() -> Type[Any]:
    """Import and return :class:`FlyRigLoaderDataSet` when Kedro is installed."""

    global _KEDRO_IMPORT_ERROR

    if _find_kedro_spec() is None:
        _KEDRO_IMPORT_ERROR = ModuleNotFoundError(
            "kedro is not installed; FlyRigLoader Kedro integration is unavailable."
        )
        raise _KEDRO_IMPORT_ERROR

    try:
        from flyrigloader.kedro.datasets import FlyRigLoaderDataSet  # type: ignore import
    except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
        _KEDRO_IMPORT_ERROR = exc
        raise

    return FlyRigLoaderDataSet


def get_dataset_class(optional: bool = False) -> Optional[Type[Any]]:
    """Return the Kedro dataset class if available."""

    try:
        return _import_dataset_class()
    except ModuleNotFoundError:
        if optional:
            return None
        message = (
            "Kedro integration requires the 'kedro' package. "
            "Install flyrigloader with the 'kedro' extra or add kedro to your environment."
        )
        if _KEDRO_IMPORT_ERROR is not None:
            logger.error(f"Kedro integration unavailable: {_KEDRO_IMPORT_ERROR}")
        raise FlyRigLoaderError(message) from _KEDRO_IMPORT_ERROR


def check_kedro_available() -> None:
    """Fail fast if Kedro is not available."""

    get_dataset_class(optional=False)


def create_kedro_dataset(*args: Any, **kwargs: Any) -> Any:
    """Factory helper that instantiates :class:`FlyRigLoaderDataSet`."""

    dataset_cls = get_dataset_class(optional=False)
    return dataset_cls(*args, **kwargs)


FlyRigLoaderDataSet: Optional[Type[Any]] = get_dataset_class(optional=True)

__all__ = [
    "FlyRigLoaderDataSet",
    "check_kedro_available",
    "create_kedro_dataset",
    "get_dataset_class",
]
