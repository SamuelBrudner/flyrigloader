"""Tests enforcing public API configuration source validation contracts."""

from pathlib import Path
from typing import Any, Callable, Dict

import pytest

from flyrigloader import api


CONFIG_SOURCE_ERROR = "Exactly one of 'config_path' or 'config' must be provided"


def _dummy_config() -> Dict[str, Any]:
    return {
        "project": {"directories": {"major_data_directory": "/tmp"}},
        "experiments": {"exp": {"datasets": ["ds"]}},
        "datasets": {"ds": {"patterns": ["*.dat"]}},
    }


@pytest.mark.parametrize(
    "api_call, kwargs",
    [
        pytest.param(
            api.discover_experiment_manifest,
            {
                "experiment_name": "exp",
                "config": _dummy_config(),
                "config_path": Path("/tmp/config.yaml"),
            },
            id="discover_manifest_with_both_sources",
        ),
        pytest.param(
            api.load_experiment_files,
            {
                "experiment_name": "exp",
                "config": _dummy_config(),
                "config_path": Path("/tmp/config.yaml"),
            },
            id="load_experiment_files_with_both_sources",
        ),
        pytest.param(
            api.load_dataset_files,
            {
                "dataset_name": "ds",
                "config": _dummy_config(),
                "config_path": Path("/tmp/config.yaml"),
            },
            id="load_dataset_files_with_both_sources",
        ),
    ],
)
def test_api_calls_reject_multiple_config_sources(
    api_call: Callable[..., Any], kwargs: Dict[str, Any]
) -> None:
    """All public API calls must reject receiving both config and config_path."""
    with pytest.raises(ValueError, match=CONFIG_SOURCE_ERROR):
        api_call(**kwargs)


@pytest.mark.parametrize(
    "api_call, kwargs",
    [
        pytest.param(
            api.discover_experiment_manifest,
            {"experiment_name": "exp"},
            id="discover_manifest_without_source",
        ),
        pytest.param(
            api.load_experiment_files,
            {"experiment_name": "exp"},
            id="load_experiment_files_without_source",
        ),
        pytest.param(
            api.load_dataset_files,
            {"dataset_name": "ds"},
            id="load_dataset_files_without_source",
        ),
    ],
)
def test_api_calls_require_configuration_source(
    api_call: Callable[..., Any], kwargs: Dict[str, Any]
) -> None:
    """All public API calls must require exactly one configuration source."""
    with pytest.raises(ValueError, match=CONFIG_SOURCE_ERROR):
        api_call(**kwargs)
