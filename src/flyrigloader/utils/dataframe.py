"""
DataFrame utilities for working with discovery results.

Utilities for converting file discovery results into pandas DataFrames
for easier analysis and manipulation.
"""

import contextlib
from typing import Union, List, Dict, Any, Optional
from pathlib import Path

import pandas as pd

from flyrigloader.discovery.stats import get_file_stats


def build_manifest_df(
    files: Union[List[str], Dict[str, Dict[str, Any]]],
    include_stats: bool = False,
    base_directory: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Convert discovery results to a pandas DataFrame.
    
    Args:
        files: Discovery results (either list of paths or dict with metadata)
        include_stats: Whether to include file statistics (size, mtime, ctime)
        base_directory: Optional base directory for calculating relative paths
        
    Returns:
        DataFrame with file information and metadata
    """
    # Handle list of files
    if isinstance(files, list):
        if not files:
            return pd.DataFrame(columns=["path"])

        # Create a DataFrame with file paths
        df = pd.DataFrame({"path": files})

        # Add file stats if requested
        if include_stats:
            stats_dfs = []
            for file_path in files:
                try:
                    stats = get_file_stats(file_path)
                    stats["path"] = file_path
                    stats_dfs.append(pd.DataFrame([stats]))
                except (FileNotFoundError, PermissionError):
                    # Skip files that don't exist or can't be accessed
                    continue

            if stats_dfs:
                stats_df = pd.concat(stats_dfs, ignore_index=True)
                df = pd.merge(df, stats_df, on="path", how="left")

        # Add relative paths if base_directory is specified
        if base_directory:
            df["relative_path"] = df["path"].apply(
                lambda p: str(Path(p).relative_to(base_directory))
            )

        return df

    # Handle dictionary with metadata
    if not files:
        return pd.DataFrame(columns=["path"])

    # Create a list of records
    records = []
    for file_path, metadata in files.items():
        record = {"path": file_path, **metadata}

        # Add file stats if requested and not already present
        if include_stats and all(
            key not in metadata for key in ["size", "mtime", "ctime"]
        ):
            with contextlib.suppress(FileNotFoundError, PermissionError):
                record |= get_file_stats(file_path)
        # Add relative path if base_directory is specified
        if base_directory:
            record["relative_path"] = str(Path(file_path).relative_to(base_directory))

        records.append(record)

    # Create DataFrame from records
    return pd.DataFrame(records)


def filter_manifest_df(
    df: pd.DataFrame,
    **filters: Any
) -> pd.DataFrame:
    """
    Filter a manifest DataFrame based on column values.
    
    Args:
        df: DataFrame to filter
        **filters: Column-value pairs for filtering (e.g., animal='mouse', condition='test')
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    for column, value in filters.items():
        if column in filtered_df.columns:
            if isinstance(value, list):
                filtered_df = filtered_df[filtered_df[column].isin(value)]
            else:
                filtered_df = filtered_df[filtered_df[column] == value]
    
    return filtered_df


def extract_unique_values(
    df: pd.DataFrame,
    column: str
) -> List[Any]:
    """
    Extract unique values from a column in a manifest DataFrame.
    
    Args:
        df: DataFrame to extract from
        column: Column name to get unique values from
        
    Returns:
        List of unique values in the column
    """
    return df[column].dropna().unique().tolist() if column in df.columns else []
