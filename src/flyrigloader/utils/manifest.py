"""Manifest-building utilities for flyrigloader.

This module gathers generic helpers that were previously scattered in
*client* repositories so that they are available directly from
``flyrigloader``.  They are **pure utilities** – completely independent of
Kedro or any other orchestration layer – and therefore safe to reuse across
projects.

Public helpers (exported via ``__all__``)
========================================
* ``attach_file_stats`` – enrich path records with basic ``os.stat`` info.
* ``build_manifest_df`` – thin re‐export of the existing DataFrame helper.
* ``build_file_manifest`` – high-level one-stop function requested in the
  feature spec.

Internal helpers (leading underscore)
======================================
* ``_convert_paths_to_absolute`` – convenience path normaliser.
* ``_load_experiment_files`` / ``_load_dataset_files`` – wrappers around
  :pymeth:`flyrigloader.api.load_experiment_files` /
  :pymeth:`flyrigloader.api.load_dataset_files` so callers don’t have to
  import from ``flyrigloader.api``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Union, Optional, Sequence

import pandas as pd

from flyrigloader import logger
from flyrigloader.utils.paths import get_absolute_path, get_relative_path
from flyrigloader.utils.dataframe import build_manifest_df as _build_manifest_df
from flyrigloader.api import (
    load_experiment_files as _api_load_experiment_files,
    load_dataset_files as _api_load_dataset_files,
)

__all__ = [
    "attach_file_stats",
    "build_manifest_df",
    "build_file_manifest",
]

PathLike = Union[str, Path]


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _convert_paths_to_absolute(paths: Sequence[PathLike]) -> List[Path]:
    """Return *absolute* :class:`~pathlib.Path` objects for *paths*."""
    abs_paths: List[Path] = []
    for p in paths:
        try:
            abs_paths.append(get_absolute_path(p, base_dir="."))  # resolves and expands
        except Exception as exc:
            logger.warning(f"Failed to resolve path {p}: {exc}")
    return abs_paths


def _load_experiment_files(
    *,
    config: Dict[str, Any] | str,
    experiment_name: str,
    base_directory: PathLike | None = None,
    extract_metadata: bool = False,
    parse_dates: bool = False,
) -> Union[List[str], Dict[str, Dict[str, Any]]]:
    """Delegate to :pyfunc:`flyrigloader.api.load_experiment_files` while
    allowing callers to request metadata extraction.
    """
    return _api_load_experiment_files(
        config_path=config if isinstance(config, str) else None,
        config=None if isinstance(config, str) else config,
        experiment_name=experiment_name,
        base_directory=base_directory,
        pattern="exp_matrix.pklz",  # enforce spec-required filtering
        recursive=True,
        extensions=["pklz"],
        extract_metadata=extract_metadata,
        parse_dates=parse_dates,
    )  # type: ignore[arg-type]


def _load_dataset_files(
    *,
    config: Dict[str, Any] | str,
    dataset_name: str,
    base_directory: PathLike | None = None,
    extract_metadata: bool = False,
    parse_dates: bool = False,
) -> Union[List[str], Dict[str, Dict[str, Any]]]:
    """Delegate to :pyfunc:`flyrigloader.api.load_dataset_files` while
    allowing callers to request metadata extraction.
    """
    return _api_load_dataset_files(
        config_path=config if isinstance(config, str) else None,
        config=None if isinstance(config, str) else config,
        dataset_name=dataset_name,
        base_directory=base_directory,
        pattern="exp_matrix.pklz",  # enforce spec-required filtering
        recursive=True,
        extensions=["pklz"],
        extract_metadata=extract_metadata,
        parse_dates=parse_dates,
    )  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# Public helpers
# -----------------------------------------------------------------------------

def attach_file_stats(
    records: List[Dict[str, Any]],
    *,
    add_timestamps: bool = True,
    add_file_size: bool = True,
) -> List[Dict[str, Any]]:
    """Attach ``os.stat`` information to every record in *records*.

    Each *record* **must** carry a key ``"path"`` holding the absolute (or
    at least resolvable) file path.
    """
    enriched: List[Dict[str, Any]] = []
    for rec in records:
        path = Path(rec["path"]).expanduser()
        try:
            stat = path.stat()
        except FileNotFoundError:
            logger.warning(f"File not found when collecting stats: {path}")
            enriched.append(rec)
            continue

        if add_file_size:
            rec["file_size"] = stat.st_size
        if add_timestamps:
            rec["mtime"] = stat.st_mtime
            rec["ctime"] = stat.st_ctime
        enriched.append(rec)
    return enriched


# We simply expose the already-existing implementation so that external callers
# have a stable import path under ``flyrigloader.utils.manifest``.
build_manifest_df = _build_manifest_df  # noqa: E305


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------

def build_file_manifest(
    config: Dict[str, Any] | str,
    experiment_name: str | None = None,
    dataset_names: list[str] | None = None,
    *,
    extract_metadata: bool = False,
    parse_dates: bool = False,
    add_timestamps: bool = True,
    add_file_size: bool = True,
    store_relative_paths: bool = True,
    base_directory: PathLike | None = None,
) -> pd.DataFrame:
    """Return a tidy :class:`pandas.DataFrame` describing experiment/dataset files.

    Parameters
    ----------
    config
        Either a path to a ``.yml`` config file or the parsed config dictionary.
    experiment_name
        Name of a *single* experiment whose files should be listed.
    dataset_names
        List of dataset identifiers.  Mutually exclusive with *experiment_name*.
    add_timestamps, add_file_size
        Whether to include ``mtime`` / ``ctime`` and ``file_size`` columns.
    store_relative_paths
        Store the ``path`` column relative to *base_directory* (if given) –
        otherwise relative to the common prefix.
    base_directory
        Optional explicit base directory used for relative path calculation.
    """
    if bool(experiment_name) == bool(dataset_names):
        raise ValueError("Specify *either* experiment_name or dataset_names, but not both.")

    # ------------------------------------------------------------------
    # 1. Collect file paths or metadata records
    # ------------------------------------------------------------------
    if experiment_name:
        file_records = _load_experiment_files(
            config=config,
            experiment_name=experiment_name,
            base_directory=base_directory,
            extract_metadata=extract_metadata,
            parse_dates=parse_dates,
        )
    else:
        assert dataset_names is not None  # mypy helper
        file_records = {} if extract_metadata else []  # type: ignore[assignment]
        for ds in dataset_names:
            result = _load_dataset_files(
                config=config,
                dataset_name=ds,
                base_directory=base_directory,
                extract_metadata=extract_metadata,
                parse_dates=parse_dates,
            )
            if extract_metadata:
                assert isinstance(result, dict)
                file_records.update(result)  # type: ignore[arg-type]
            else:
                assert isinstance(result, list)
                file_records.extend(result)  # type: ignore[arg-type]

    logger.info(f"Collected {len(file_records)} exp_matrix.pklz files.")

    # ------------------------------------------------------------------
    # 2. Convert to DataFrame (initial cols: path, plus whatever metadata)
    # ------------------------------------------------------------------
    df = build_manifest_df(file_records, include_stats=False)

    # ------------------------------------------------------------------
    # 3. Optionally enrich with os.stat info
    # ------------------------------------------------------------------
    if add_timestamps or add_file_size:
        records = df.to_dict(orient="records")
        records = attach_file_stats(
            records,
            add_timestamps=add_timestamps,
            add_file_size=add_file_size,
        )
        df = pd.DataFrame.from_records(records)

    # ------------------------------------------------------------------
    # 4. Re-encode paths relative to *base_directory* if requested
    # ------------------------------------------------------------------
    if store_relative_paths:
        # Determine base_dir default lazily if not provided.
        if base_directory is None:
            # Use common parent among all files.
            path_list = list(file_records.keys()) if isinstance(file_records, dict) else file_records  # type: ignore[arg-type]
            common_parts = os.path.commonpath(path_list)
            base_directory = common_parts
        df["path"] = df["path"].apply(lambda p: str(get_relative_path(p, base_directory)))

    return df
