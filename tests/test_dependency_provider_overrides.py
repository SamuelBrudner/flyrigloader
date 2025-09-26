"""Tests for dependency provider override behavior."""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Union

from flyrigloader.api.dependencies import (
    AbstractDependencyProvider,
    ConfigProvider,
    DiscoveryProvider,
    IOProvider,
    UtilsProvider,
    get_dependency_provider,
    reset_dependency_provider,
    set_dependency_provider,
    use_dependency_provider,
)


class _FakeDependencyProvider(AbstractDependencyProvider):
    """A lightweight provider used for testing injection hooks."""

    class _Config(ConfigProvider):
        def load_config(self, config_path: Union[str, Any]) -> Dict[str, Any]:  # pragma: no cover - unused
            return {"config_path": str(config_path)}

        def get_ignore_patterns(
            self,
            config: Union[Dict[str, Any], Any],
            experiment: Optional[str] = None,
        ) -> List[str]:  # pragma: no cover - unused
            return []

        def get_mandatory_substrings(
            self,
            config: Union[Dict[str, Any], Any],
            experiment: Optional[str] = None,
        ) -> List[str]:  # pragma: no cover - unused
            return []

        def get_dataset_info(
            self, config: Union[Dict[str, Any], Any], dataset_name: str
        ) -> Dict[str, Any]:  # pragma: no cover - unused
            return {"dataset": dataset_name}

        def get_experiment_info(
            self, config: Union[Dict[str, Any], Any], experiment_name: str
        ) -> Dict[str, Any]:  # pragma: no cover - unused
            return {"experiment": experiment_name}

    class _Discovery(DiscoveryProvider):
        def discover_files_with_config(
            self,
            config: Union[Dict[str, Any], Any],
            directory: Union[str, List[str]],
            pattern: str,
            recursive: bool = False,
            extensions: Optional[List[str]] = None,
            experiment: Optional[str] = None,
            extract_metadata: bool = False,
            parse_dates: bool = False,
        ) -> Union[List[str], Dict[str, Dict[str, Any]]]:  # pragma: no cover - unused
            return []

        def discover_experiment_files(
            self,
            config: Union[Dict[str, Any], Any],
            experiment_name: str,
            base_directory: Union[str, Any],
            pattern: str = "*.*",
            recursive: bool = True,
            extensions: Optional[List[str]] = None,
            extract_metadata: bool = False,
            parse_dates: bool = False,
        ) -> Union[List[str], Dict[str, Dict[str, Any]]]:  # pragma: no cover - unused
            return []

        def discover_dataset_files(
            self,
            config: Union[Dict[str, Any], Any],
            dataset_name: str,
            base_directory: Union[str, Any],
            pattern: str = "*.*",
            recursive: bool = True,
            extensions: Optional[List[str]] = None,
            extract_metadata: bool = False,
            parse_dates: bool = False,
        ) -> Union[List[str], Dict[str, Dict[str, Any]]]:  # pragma: no cover - unused
            return []

    class _IO(IOProvider):
        def read_pickle_any_format(self, path: Union[str, Any]) -> Any:  # pragma: no cover - unused
            return path

        def make_dataframe_from_config(
            self,
            exp_matrix: Dict[str, Any],
            config_source: Optional[Union[str, Any, Dict[str, Any]]] = None,
            metadata: Optional[Dict[str, Any]] = None,
        ) -> Any:  # pragma: no cover - unused
            return exp_matrix

        def get_config_from_source(
            self, config_source: Optional[Union[str, Any, Dict[str, Any]]] = None
        ) -> Any:  # pragma: no cover - unused
            return config_source

    class _Utils(UtilsProvider):
        def get_file_stats(self, path: Union[str, Any]) -> Dict[str, Any]:  # pragma: no cover - unused
            return {"path": path}

        def get_relative_path(self, path: Union[str, Any], base_dir: Union[str, Any]) -> Any:  # pragma: no cover - unused
            return path

        def get_absolute_path(self, path: Union[str, Any], base_dir: Union[str, Any]) -> Any:  # pragma: no cover - unused
            return path

        def check_file_exists(self, path: Union[str, Any]) -> bool:  # pragma: no cover - unused
            return True

        def ensure_directory_exists(self, path: Union[str, Any]) -> Any:  # pragma: no cover - unused
            return path

        def find_common_base_directory(
            self, paths: List[Union[str, Any]]
        ) -> Optional[Any]:  # pragma: no cover - unused
            return None

    def __init__(self) -> None:
        self._config = self._Config()
        self._discovery = self._Discovery()
        self._io = self._IO()
        self._utils = self._Utils()

    @property
    def config(self) -> ConfigProvider:
        return self._config

    @property
    def discovery(self) -> DiscoveryProvider:
        return self._discovery

    @property
    def io(self) -> IOProvider:
        return self._io

    @property
    def utils(self) -> UtilsProvider:
        return self._utils


def _prepare_default_provider() -> Any:
    reset_dependency_provider()
    return get_dependency_provider()


def test_set_dependency_provider_accepts_custom_implementations() -> None:
    default_provider = _prepare_default_provider()
    fake_provider = _FakeDependencyProvider()

    try:
        set_dependency_provider(fake_provider)
        assert get_dependency_provider() is fake_provider
    finally:
        set_dependency_provider(default_provider)
        reset_dependency_provider()


def test_use_dependency_provider_is_isolated_per_thread() -> None:
    default_provider = _prepare_default_provider()
    fake_provider = _FakeDependencyProvider()

    observed: list[Any] = []

    def capture_provider() -> None:
        observed.append(get_dependency_provider())

    try:
        with use_dependency_provider(fake_provider):
            thread = threading.Thread(target=capture_provider)
            thread.start()
            thread.join()

        assert len(observed) == 1
        thread_provider = observed[0]
        assert isinstance(thread_provider, type(default_provider))
        assert thread_provider is not fake_provider
    finally:
        reset_dependency_provider()

