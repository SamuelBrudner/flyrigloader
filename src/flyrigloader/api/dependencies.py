"""Dependency provider infrastructure for :mod:`flyrigloader.api`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Protocol, Union

from flyrigloader import logger
from flyrigloader.config.models import LegacyConfigAdapter
from flyrigloader.io.column_models import get_config_from_source


class ConfigProvider(Protocol):
    """Protocol for configuration providers supporting dependency injection."""

    def load_config(self, config_path: Union[str, Path]) -> Union[Dict[str, Any], LegacyConfigAdapter]:
        ...

    def get_ignore_patterns(
        self,
        config: Union[Dict[str, Any], LegacyConfigAdapter],
        experiment: Optional[str] = None,
    ) -> List[str]:
        ...

    def get_mandatory_substrings(
        self,
        config: Union[Dict[str, Any], LegacyConfigAdapter],
        experiment: Optional[str] = None,
    ) -> List[str]:
        ...

    def get_dataset_info(
        self,
        config: Union[Dict[str, Any], LegacyConfigAdapter],
        dataset_name: str,
    ) -> Dict[str, Any]:
        ...

    def get_experiment_info(
        self,
        config: Union[Dict[str, Any], LegacyConfigAdapter],
        experiment_name: str,
    ) -> Dict[str, Any]:
        ...


class DiscoveryProvider(Protocol):
    """Protocol for file discovery providers supporting dependency injection."""

    def discover_files_with_config(
        self,
        config: Union[Dict[str, Any], LegacyConfigAdapter],
        directory: Union[str, List[str]],
        pattern: str,
        recursive: bool = False,
        extensions: Optional[List[str]] = None,
        experiment: Optional[str] = None,
        extract_metadata: bool = False,
        parse_dates: bool = False,
    ) -> Union[List[str], Dict[str, Dict[str, Any]]]:
        ...

    def discover_experiment_files(
        self,
        config: Union[Dict[str, Any], LegacyConfigAdapter],
        experiment_name: str,
        base_directory: Union[str, Path],
        pattern: str = "*.*",
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
        extract_metadata: bool = False,
        parse_dates: bool = False,
    ) -> Union[List[str], Dict[str, Dict[str, Any]]]:
        ...

    def discover_dataset_files(
        self,
        config: Union[Dict[str, Any], LegacyConfigAdapter],
        dataset_name: str,
        base_directory: Union[str, Path],
        pattern: str = "*.*",
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
        extract_metadata: bool = False,
        parse_dates: bool = False,
    ) -> Union[List[str], Dict[str, Dict[str, Any]]]:
        ...


class IOProvider(Protocol):
    """Protocol for I/O providers supporting dependency injection."""

    def read_pickle_any_format(self, path: Union[str, Path]) -> Any:
        ...

    def make_dataframe_from_config(
        self,
        exp_matrix: Dict[str, Any],
        config_source: Optional[Union[str, Path, Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        ...

    def get_config_from_source(
        self,
        config_source: Optional[Union[str, Path, Dict[str, Any]]] = None,
    ) -> Any:
        ...


class UtilsProvider(Protocol):
    """Protocol for utility providers supporting dependency injection."""

    def get_file_stats(self, path: Union[str, Path]) -> Dict[str, Any]:
        ...

    def get_relative_path(self, path: Union[str, Path], base_dir: Union[str, Path]) -> Path:
        ...

    def get_absolute_path(self, path: Union[str, Path], base_dir: Union[str, Path]) -> Path:
        ...

    def check_file_exists(self, path: Union[str, Path]) -> bool:
        ...

    def ensure_directory_exists(self, path: Union[str, Path]) -> Path:
        ...

    def find_common_base_directory(self, paths: List[Union[str, Path]]) -> Optional[Path]:
        ...


class ManifestProvider(Protocol):
    """Protocol describing manifest conversion helpers used by the API."""

    def attach_metadata_bucket(
        self,
        manifest: Dict[str, Dict[str, Any]],
        metadata_bucket: Optional[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        ...


class AbstractDependencyProvider(ABC):
    """Base class for dependency providers to support dependency injection."""

    @property
    @abstractmethod
    def config(self) -> ConfigProvider:  # pragma: no cover - interface definition
        ...

    @property
    @abstractmethod
    def discovery(self) -> DiscoveryProvider:  # pragma: no cover - interface definition
        ...

    @property
    @abstractmethod
    def io(self) -> IOProvider:  # pragma: no cover - interface definition
        ...

    @property
    @abstractmethod
    def utils(self) -> UtilsProvider:  # pragma: no cover - interface definition
        ...


class DefaultDependencyProvider(AbstractDependencyProvider):
    """Default implementation of dependency providers using actual modules."""

    def __init__(self):
        self._config_module: ConfigProvider | None = None
        self._discovery_module: DiscoveryProvider | None = None
        self._io_module: IOProvider | None = None
        self._utils_module: UtilsProvider | None = None
        logger.debug("Initialized DefaultDependencyProvider with lazy loading")

    @property
    def config(self) -> ConfigProvider:
        if self._config_module is None:
            logger.debug("Loading configuration module dependencies")
            from flyrigloader.config import yaml_config as _yaml_config

            _load_config = _yaml_config.load_config
            _get_ignore_patterns = _yaml_config.get_ignore_patterns
            _get_mandatory_substrings = _yaml_config.get_mandatory_substrings
            _get_dataset_info = _yaml_config.get_dataset_info
            _get_experiment_info = _yaml_config.get_experiment_info

            class ConfigModule:
                load_config = staticmethod(_load_config)
                get_ignore_patterns = staticmethod(_get_ignore_patterns)
                get_mandatory_substrings = staticmethod(_get_mandatory_substrings)
                get_dataset_info = staticmethod(_get_dataset_info)
                get_experiment_info = staticmethod(_get_experiment_info)

            self._config_module = ConfigModule()  # type: ignore[assignment]
        return self._config_module

    @property
    def discovery(self) -> DiscoveryProvider:
        if self._discovery_module is None:
            logger.debug("Loading discovery module dependencies")
            import importlib

            discovery_mod = importlib.import_module("flyrigloader.config.discovery")

            class DiscoveryModule:
                discover_files_with_config = staticmethod(discovery_mod.discover_files_with_config)
                discover_experiment_files = staticmethod(discovery_mod.discover_experiment_files)
                discover_dataset_files = staticmethod(discovery_mod.discover_dataset_files)

            self._discovery_module = DiscoveryModule()  # type: ignore[assignment]
        return self._discovery_module

    @property
    def io(self) -> IOProvider:
        if self._io_module is None:
            logger.debug("Loading I/O module dependencies")
            from flyrigloader.io.pickle import read_pickle_any_format
            from flyrigloader.io.transformers import make_dataframe_from_config
            from types import SimpleNamespace

            self._io_module = SimpleNamespace(
                read_pickle_any_format=read_pickle_any_format,
                make_dataframe_from_config=make_dataframe_from_config,
                get_config_from_source=get_config_from_source,
            )
        return self._io_module

    @property
    def utils(self) -> UtilsProvider:
        if self._utils_module is None:
            logger.debug("Loading utilities module dependencies")
            from flyrigloader.discovery.stats import get_file_stats as _get_file_stats
            from flyrigloader.utils.paths import (
                check_file_exists as _check_file_exists,
                ensure_directory_exists as _ensure_directory_exists,
                find_common_base_directory as _find_common_base_directory,
                get_absolute_path as _get_absolute_path,
                get_relative_path as _get_relative_path,
            )

            class UtilsModule:
                get_file_stats = staticmethod(_get_file_stats)
                get_relative_path = staticmethod(_get_relative_path)
                get_absolute_path = staticmethod(_get_absolute_path)
                check_file_exists = staticmethod(_check_file_exists)
                ensure_directory_exists = staticmethod(_ensure_directory_exists)
                find_common_base_directory = staticmethod(_find_common_base_directory)

            self._utils_module = UtilsModule()  # type: ignore[assignment]
        return self._utils_module


class _DependencyProviderState:
    """Thread-safe registry for the active dependency provider."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._provider: DefaultDependencyProvider = DefaultDependencyProvider()
        self._override_stack: list[DefaultDependencyProvider] = []

    def get(self) -> DefaultDependencyProvider:
        with self._lock:
            return self._provider

    def set(self, provider: DefaultDependencyProvider) -> None:
        with self._lock:
            self._provider = provider

    def reset(self) -> None:
        with self._lock:
            self._provider = DefaultDependencyProvider()
            self._override_stack.clear()

    def push_override(self, provider: DefaultDependencyProvider) -> None:
        with self._lock:
            logger.debug("Setting dependency provider to %s", type(provider).__name__)
            self._override_stack.append(self._provider)
            self._provider = provider

    def pop_override(self, expected: DefaultDependencyProvider) -> DefaultDependencyProvider:
        with self._lock:
            if not self._override_stack:
                logger.debug(
                    "Override stack empty; keeping provider %s", type(self._provider).__name__
                )
                return self._provider

            previous = self._override_stack.pop()
            if self._provider is expected:
                logger.debug(
                    "Restoring dependency provider to %s", type(previous).__name__
                )
                self._provider = previous
            else:
                logger.debug(
                    "Dependency provider changed while override active; keeping %s",
                    type(self._provider).__name__,
                )
            return self._provider


_dependency_state = _DependencyProviderState()


def set_dependency_provider(provider: DefaultDependencyProvider) -> None:
    """Set the global dependency provider for testing purposes."""

    if not isinstance(provider, DefaultDependencyProvider):  # pragma: no cover - defensive branch
        raise TypeError("provider must be an instance of DefaultDependencyProvider")

    logger.debug("Setting dependency provider to %s", type(provider).__name__)
    _dependency_state.set(provider)


def get_dependency_provider() -> DefaultDependencyProvider:
    """Return the current dependency provider."""

    return _dependency_state.get()


def reset_dependency_provider() -> None:
    """Reset the dependency provider to the default implementation."""

    logger.debug("Resetting dependency provider to default")
    _dependency_state.reset()


@contextmanager
def use_dependency_provider(
    provider: DefaultDependencyProvider,
) -> Iterator[DefaultDependencyProvider]:
    """Temporarily override the active dependency provider.

    Parameters
    ----------
    provider:
        The provider instance to activate within the context block.

    Yields
    ------
    DefaultDependencyProvider
        The provider that was supplied, allowing callers to inspect or
        configure it further while the override is active.
    """

    if provider is None:
        raise ValueError("provider must not be None")
    if not isinstance(provider, DefaultDependencyProvider):
        raise TypeError("provider must be an instance of DefaultDependencyProvider")

    previous_provider = get_dependency_provider()
    logger.debug(
        "Entering dependency provider override: %s -> %s",
        type(previous_provider).__name__,
        type(provider).__name__,
    )
    _dependency_state.push_override(provider)

    try:
        yield provider
    finally:
        logger.debug(
            "Restoring dependency provider to %s", type(previous_provider).__name__
        )
        _dependency_state.pop_override(provider)


__all__ = [
    "AbstractDependencyProvider",
    "ConfigProvider",
    "DefaultDependencyProvider",
    "DiscoveryProvider",
    "IOProvider",
    "ManifestProvider",
    "UtilsProvider",
    "get_dependency_provider",
    "reset_dependency_provider",
    "set_dependency_provider",
    "use_dependency_provider",
]
