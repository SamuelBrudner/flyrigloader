"""Smoke tests ensuring DefaultDependencyProvider exposes critical IO helpers.

These tests guard against regressions where `make_dataframe_from_config` or
`get_config_from_source` are not available via the lazily-loaded `io` module.
"""

from flyrigloader.api import get_dependency_provider


def test_io_exports() -> None:
    """`dp.io` should provide the expected helper callables."""
    dp = get_dependency_provider()

    # The attributes must exist and be callables
    assert hasattr(dp.io, "make_dataframe_from_config"), "Missing make_dataframe_from_config on dp.io"
    assert hasattr(dp.io, "get_config_from_source"), "Missing get_config_from_source on dp.io"

    assert callable(dp.io.make_dataframe_from_config)
    assert callable(dp.io.get_config_from_source)
