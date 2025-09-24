"""Shared helper utilities for the public API modules."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from flyrigloader import logger

from .dependencies import (
    ConfigProvider,
    DefaultDependencyProvider,
    DiscoveryProvider,
    IOProvider,
    UtilsProvider,
)

def _attach_metadata_bucket(
    discovery_result: Union[List[str], Dict[str, Any]]
) -> Union[List[str], Dict[str, Dict[str, Any]]]:
    """Ensure discovery results provide a nested ``metadata`` dictionary."""

    if not isinstance(discovery_result, dict):
        return discovery_result

    normalised: Dict[str, Dict[str, Any]] = {}

    for path, payload in discovery_result.items():
        path_str = str(path)

        if not isinstance(payload, dict):
            normalised[path_str] = {"path": path_str, "metadata": {}}
            continue

        flattened = dict(payload)
        flattened.setdefault("path", path_str)

        existing_metadata = flattened.get("metadata")
        metadata_bucket: Dict[str, Any] = {}
        if isinstance(existing_metadata, dict):
            metadata_bucket.update(existing_metadata)

        for key, value in flattened.items():
            if key in {"metadata", "path"}:
                continue
            metadata_bucket.setdefault(key, value)

        flattened["metadata"] = metadata_bucket
        normalised[path_str] = flattened

    return normalised

def _create_test_dependency_provider(
    config_provider: Optional[ConfigProvider] = None,
    discovery_provider: Optional[DiscoveryProvider] = None,
    io_provider: Optional[IOProvider] = None,
    utils_provider: Optional[UtilsProvider] = None
) -> DefaultDependencyProvider:
    """
    Create a test dependency provider with optional mock providers.
    
    This function supports comprehensive testing scenarios by allowing individual
    providers to be mocked while maintaining the overall dependency structure.
    
    Args:
        config_provider: Optional mock configuration provider
        discovery_provider: Optional mock discovery provider
        io_provider: Optional mock I/O provider
        utils_provider: Optional mock utilities provider
        
    Returns:
        DefaultDependencyProvider instance configured for testing
        
    Example:
        >>> from unittest.mock import Mock
        >>> mock_config = Mock(spec=ConfigProvider)
        >>> test_deps = _create_test_dependency_provider(config_provider=mock_config)
        >>> set_dependency_provider(test_deps)
    """
    logger.debug("Creating test dependency provider")
    
    test_provider = DefaultDependencyProvider()
    
    # Override individual providers if specified
    if config_provider is not None:
        test_provider._config_module = config_provider
        logger.debug("Injected custom config provider")
    if discovery_provider is not None:
        test_provider._discovery_module = discovery_provider
        logger.debug("Injected custom discovery provider")
    if io_provider is not None:
        test_provider._io_module = io_provider
        logger.debug("Injected custom I/O provider")
    if utils_provider is not None:
        test_provider._utils_module = utils_provider
        logger.debug("Injected custom utils provider")
    
    return test_provider
