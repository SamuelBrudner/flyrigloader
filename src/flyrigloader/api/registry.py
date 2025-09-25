"""Registry introspection helpers for :mod:`flyrigloader.api`."""

from __future__ import annotations

from typing import Any, Dict, Optional

from flyrigloader import logger
from flyrigloader.exceptions import FlyRigLoaderError
from flyrigloader.registries import get_loader_capabilities as _get_loader_capabilities

from .dependencies import DefaultDependencyProvider, get_dependency_provider


def get_registered_loaders(
    _deps: Optional[DefaultDependencyProvider] = None,
) -> Dict[str, Dict[str, Any]]:
    """Return comprehensive information about registered file loaders."""

    operation_name = "get_registered_loaders"

    if _deps is None:
        _deps = get_dependency_provider()

    logger.debug("🔍 Retrieving registered loader information")

    try:
        from flyrigloader.registries import LoaderRegistry

        registry = LoaderRegistry()
        loader_info: Dict[str, Dict[str, Any]] = {}

        all_loaders = registry.get_all_loaders()
        registered_extensions = list(all_loaders.keys())
        logger.debug("Found %s registered extensions", len(registered_extensions))

        for extension in registered_extensions:
            try:
                loader_class = registry.get_loader_for_extension(extension)
                capabilities = _get_loader_capabilities(extension)

                loader_info[extension] = {
                    "loader_class": loader_class.__name__ if hasattr(loader_class, "__name__") else str(loader_class),
                    "supported_extensions": [extension],
                    "priority": getattr(loader_class, "priority", "BUILTIN"),
                    "capabilities": capabilities,
                    "metadata": {
                        "module": getattr(loader_class, "__module__", "unknown"),
                        "registered": True,
                        "extension_primary": extension,
                    },
                }

                if hasattr(loader_class, "supported_extensions"):
                    additional_extensions = [
                        ext for ext in loader_class.supported_extensions if ext != extension and ext in registered_extensions
                    ]
                    if additional_extensions:
                        loader_info[extension]["supported_extensions"].extend(additional_extensions)

                logger.debug("Retrieved information for loader: %s", extension)

            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to get information for extension %s: %s", extension, exc)
                loader_info[extension] = {
                    "loader_class": "unknown",
                    "supported_extensions": [extension],
                    "priority": "unknown",
                    "capabilities": {},
                    "metadata": {
                        "error": str(exc),
                        "registered": True,
                        "extension_primary": extension,
                    },
                }

        logger.info("✓ Retrieved information for %s registered loaders", len(loader_info))
        logger.debug("  Extensions: %s", list(loader_info.keys()))

        return loader_info

    except Exception as exc:
        error_msg = (
            f"Failed to retrieve registered loaders for {operation_name}: {exc}. "
            "Please check the registry system and loader registrations."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from exc


def get_loader_capabilities(
    extension: Optional[str] = None,
    _deps: Optional[DefaultDependencyProvider] = None,
) -> Dict[str, Any]:
    """Return capability metadata for registered file loaders."""

    operation_name = "get_loader_capabilities"

    if _deps is None:
        _deps = get_dependency_provider()

    logger.debug("🔍 Retrieving loader capabilities for extension: %s", extension or "all")

    try:
        if extension is not None:
            if not extension or not isinstance(extension, str):
                error_msg = (
                    f"Invalid extension for {operation_name}: '{extension}'. "
                    "extension must be a non-empty string (e.g., '.pkl')."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            capabilities = _get_loader_capabilities(extension)

            if not capabilities:
                error_msg = (
                    f"No loader registered for extension '{extension}'. "
                    "Please check the extension format and ensure a loader is registered."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.debug("Retrieved capabilities for extension %s", extension)
            return capabilities

        from flyrigloader.registries import LoaderRegistry

        registry = LoaderRegistry()
        all_loaders = registry.get_all_loaders()
        registered_extensions = list(all_loaders.keys())

        all_capabilities: Dict[str, Dict[str, Any]] = {}
        for ext in registered_extensions:
            try:
                capabilities = _get_loader_capabilities(ext)
                all_capabilities[ext] = capabilities
                logger.debug("Retrieved capabilities for extension %s", ext)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to get capabilities for extension %s: %s", ext, exc)
                all_capabilities[ext] = {
                    "streaming_support": False,
                    "compression_support": [],
                    "metadata_extraction": False,
                    "performance_profile": {"status": "unknown"},
                    "memory_efficiency": {"rating": "unknown"},
                    "thread_safety": {"safe": False},
                    "error": str(exc),
                }

        logger.info("✓ Retrieved capabilities for %s loaders", len(all_capabilities))
        return all_capabilities

    except Exception as exc:
        if isinstance(exc, ValueError):
            raise

        error_msg = (
            f"Failed to retrieve loader capabilities for {operation_name}: {exc}. "
            "Please check the registry system and loader implementations."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from exc


__all__ = ["get_loader_capabilities", "get_registered_loaders"]

