"""Data transformation helpers for :mod:`flyrigloader.api`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import pandas as pd

from flyrigloader import logger
from flyrigloader.exceptions import FlyRigLoaderError
from flyrigloader.io.column_models import get_config_from_source as _get_config_from_source
from flyrigloader.io.transformers import transform_to_dataframe as _transform_to_dataframe

from ._shared import resolve_api_override
from .dependencies import DefaultDependencyProvider, get_dependency_provider


def _get_api_override(name: str, fallback: Callable[..., Any]) -> Callable[..., Any]:
    """Return patched attribute from :mod:`flyrigloader.api` when available."""

    return resolve_api_override(globals(), name, fallback)


def transform_to_dataframe(
    raw_data: Dict[str, Any],
    column_config_path: Optional[Union[str, Path, Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    add_file_path: bool = True,
    file_path: Optional[Union[str, Path]] = None,
    strict_schema: bool = False,
    _deps: Optional[DefaultDependencyProvider] = None,
) -> pd.DataFrame:
    """Transform raw experimental data into a pandas ``DataFrame``."""

    operation_name = "transform_to_dataframe"

    if _deps is None:
        _deps = get_dependency_provider()

    logger.debug("🔄 Transforming raw data to DataFrame")

    if not isinstance(raw_data, dict):
        error_msg = (
            f"Invalid raw_data for {operation_name}: expected dict, got {type(raw_data).__name__}. "
            "raw_data must be a dictionary containing experimental data columns."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)

    if not raw_data:
        error_msg = (
            f"Empty raw_data for {operation_name}. "
            "raw_data must contain at least one data column."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)

    if add_file_path and not file_path:
        error_msg = (
            f"file_path parameter required when add_file_path=True for {operation_name}. "
            "Please provide the source file path or set add_file_path=False."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)

    logger.debug("Transforming %s data columns", len(raw_data))
    if metadata:
        logger.debug("Adding metadata keys: %s", list(metadata.keys()))

    try:
        transformer = _get_api_override("_transform_to_dataframe", _transform_to_dataframe)
        df = transformer(
            exp_matrix=raw_data,
            config_source=column_config_path,
            metadata=metadata,
        )

        if add_file_path and file_path:
            df["file_path"] = str(Path(file_path).resolve())

        if strict_schema:
            if column_config_path is None:
                raise FlyRigLoaderError(
                    "strict_schema=True requires a column_config_path (schema) to be provided"
                )
            schema_model = _get_config_from_source(column_config_path)
            allowed_cols = set(schema_model.columns.keys())
            if add_file_path:
                allowed_cols.add("file_path")

            missing = [col for col in df.columns if col not in allowed_cols]
            if missing:
                logger.debug("Dropping %s columns not present in schema", missing)
                df = df[[col for col in df.columns if col in allowed_cols]]

        logger.debug("✓ Successfully transformed to DataFrame with shape: %s", df.shape)
        return df
    except FlyRigLoaderError:
        raise
    except Exception as exc:  # pragma: no cover - defensive logging
        error_msg = (
            f"Failed to transform raw data to DataFrame for {operation_name}: {exc}. "
            "Please check the data structure and column configuration compatibility."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from exc


__all__ = ["transform_to_dataframe"]
