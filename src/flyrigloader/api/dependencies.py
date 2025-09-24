"""Dependency providers used by the public API facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

from flyrigloader import logger
from flyrigloader.config.models import LegacyConfigAdapter
from flyrigloader.io.column_models import get_config_from_source


class ConfigProvider(Protocol):
    """Protocol for configuration providers supporting dependency injection."""

    def load_config(self, config_path: Union[str, Path]) -> Union[Dict[str, Any], LegacyConfigAdapter]: ...

    def get_ignore_patterns(
        self,
        config: Union[Dict[str, Any], LegacyConfigAdapter],
        experiment: Optional[str] = None,
    ) -> List[str]: ...

    def get_mandatory_substrings(
        self,
        config: Union[Dict[str, Any], LegacyConfigAdapter],
        experiment: Optional[str] = None,
    ) -> List[str]: ...

    def get_dataset_info(
        self,
        config: Union[Dict[str, Any], LegacyConfigAdapter],
        dataset_name: str,
    ) -> Dict[str, Any]: ...

    def get_experiment_info(
        self,
        config: Union[Dict[str, Any], LegacyConfigAdapter],
        experiment_name: str,
    ) -> Dict[str, Any]: ...


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
    ) -> Union[List[str], Dict[str, Dict[str, Any]]]: ...

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
    ) -> Union[List[str], Dict[str, Dict[str, Any]]]: ...

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
    ) -> Union[List[str], Dict[str, Dict[str, Any]]]: ...


class IOProvider(Protocol):
    """Protocol for I/O providers supporting dependency injection."""

    def read_pickle_any_format(self, path: Union[str, Path]) -> Any: ...

    def make_dataframe_from_config(
        self,
        data: Any,
        config: Optional[Dict[str, Any]] = None,
        config_source: Optional[Union[str, Path, Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any: ...

    def get_config_from_source(self, config_source: Optional[Union[str, Path, Dict[str, Any]]] = None) -> Any: ...


class UtilsProvider(Protocol):
    """Protocol for utility providers supporting dependency injection."""

    def get_file_stats(self, path: Union[str, Path]) -> Dict[str, Any]: ...

    def get_relative_path(self, path: Union[str, Path], base_dir: Union[str, Path]) -> Path: ...

    def get_absolute_path(self, path: Union[str, Path], base_dir: Union[str, Path]) -> Path: ...

    def check_file_exists(self, path: Union[str, Path]) -> bool: ...

    def ensure_directory_exists(self, path: Union[str, Path]) -> Path: ...

    def find_common_base_directory(self, paths: List[Union[str, Path]]) -> Optional[Path]: ...


class DefaultDependencyProvider:
    """Default implementation of dependency providers using actual modules."""

    def __init__(self) -> None:
        self._config_module: Optional[ConfigProvider] = None
        self._discovery_module: Optional[DiscoveryProvider] = None
        self._io_module: Optional[IOProvider] = None
        self._utils_module: Optional[UtilsProvider] = None
        logger.debug("Initialized DefaultDependencyProvider with lazy loading")

    @property
    def config(self) -> ConfigProvider:
        if self._config_module is None:
            logger.debug("Loading configuration module dependencies")
            from flyrigloader.config import yaml_config as _yaml_config

            class ConfigModule:
                load_config = staticmethod(_yaml_config.load_config)
                get_ignore_patterns = staticmethod(_yaml_config.get_ignore_patterns)
                get_mandatory_substrings = staticmethod(_yaml_config.get_mandatory_substrings)
                get_dataset_info = staticmethod(_yaml_config.get_dataset_info)
                get_experiment_info = staticmethod(_yaml_config.get_experiment_info)

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

            class IOModule:
                pass

            IOModule.read_pickle_any_format = staticmethod(read_pickle_any_format)
            IOModule.make_dataframe_from_config = staticmethod(make_dataframe_from_config)
            IOModule.get_config_from_source = staticmethod(get_config_from_source)

            self._io_module = IOModule()  # type: ignore[assignment]
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


_dependency_provider: DefaultDependencyProvider = DefaultDependencyProvider()


def set_dependency_provider(provider: DefaultDependencyProvider) -> None:
    logger.debug(f"Setting dependency provider to {type(provider).__name__}")
    global _dependency_provider
    _dependency_provider = provider


def get_dependency_provider() -> DefaultDependencyProvider:
    return _dependency_provider


def reset_dependency_provider() -> None:
    logger.debug("Resetting dependency provider to default")
    global _dependency_provider
    _dependency_provider = DefaultDependencyProvider()
