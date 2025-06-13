"""
DataFrame utilities for working with discovery results.

Utilities for converting file discovery results into pandas DataFrames
for easier analysis and manipulation. Enhanced with dependency injection
patterns and comprehensive error handling for improved testability.
"""

import contextlib
from typing import Union, List, Dict, Any, Optional, Protocol, Callable
from pathlib import Path
from abc import ABC, abstractmethod

import pandas as pd
from flyrigloader import logger

from flyrigloader.discovery.stats import get_file_stats
from flyrigloader.utils.paths import get_relative_path


class DataFrameProvider(Protocol):
    """Protocol for DataFrame creation to enable dependency injection."""
    
    def create_dataframe(self, data: Any, **kwargs) -> pd.DataFrame:
        """Create a DataFrame from data."""
        ...
    
    def concat_dataframes(self, dataframes: List[pd.DataFrame], **kwargs) -> pd.DataFrame:
        """Concatenate multiple DataFrames."""
        ...
    
    def merge_dataframes(self, left: pd.DataFrame, right: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Merge two DataFrames."""
        ...


class FileStatsProvider(Protocol):
    """Protocol for file statistics collection to enable dependency injection."""
    
    def get_stats(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Get file statistics for a given path."""
        ...


class PathProvider(Protocol):
    """Protocol for path operations to enable dependency injection."""
    
    def get_relative_path(self, path: Union[str, Path], base_dir: Union[str, Path]) -> Path:
        """Get relative path from base directory."""
        ...


class DefaultDataFrameProvider:
    """Default implementation of DataFrameProvider using pandas."""
    
    def create_dataframe(self, data: Any, **kwargs) -> pd.DataFrame:
        """Create a DataFrame using pandas.DataFrame constructor."""
        try:
            logger.debug(f"Creating DataFrame with data type: {type(data)}")
            return pd.DataFrame(data, **kwargs)
        except Exception as e:
            logger.error(f"Failed to create DataFrame: {e}")
            raise
    
    def concat_dataframes(self, dataframes: List[pd.DataFrame], **kwargs) -> pd.DataFrame:
        """Concatenate DataFrames using pandas.concat."""
        try:
            logger.debug(f"Concatenating {len(dataframes)} DataFrames")
            if not dataframes:
                logger.warning("No DataFrames to concatenate")
                return pd.DataFrame()
            return pd.concat(dataframes, **kwargs)
        except Exception as e:
            logger.error(f"Failed to concatenate DataFrames: {e}")
            raise
    
    def merge_dataframes(self, left: pd.DataFrame, right: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Merge DataFrames using pandas.merge."""
        try:
            logger.debug(f"Merging DataFrames - left shape: {left.shape}, right shape: {right.shape}")
            return pd.merge(left, right, **kwargs)
        except Exception as e:
            logger.error(f"Failed to merge DataFrames: {e}")
            raise


class DefaultFileStatsProvider:
    """Default implementation of FileStatsProvider using flyrigloader.discovery.stats."""
    
    def get_stats(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Get file statistics using get_file_stats function."""
        try:
            logger.debug(f"Getting file stats for: {path}")
            return get_file_stats(path)
        except Exception as e:
            logger.warning(f"Failed to get file stats for {path}: {e}")
            raise


class DefaultPathProvider:
    """Default implementation of PathProvider using flyrigloader.utils.paths."""
    
    def get_relative_path(self, path: Union[str, Path], base_dir: Union[str, Path]) -> Path:
        """Get relative path using path utilities."""
        try:
            logger.debug(f"Getting relative path for {path} from base {base_dir}")
            return get_relative_path(path, base_dir)
        except Exception as e:
            logger.warning(f"Failed to get relative path for {path}: {e}")
            # Fallback to basic pathlib relative_to
            return Path(path).relative_to(base_dir)


class DataFrameUtilities:
    """Enhanced DataFrame utilities with dependency injection support."""
    
    def __init__(
        self,
        dataframe_provider: Optional[DataFrameProvider] = None,
        file_stats_provider: Optional[FileStatsProvider] = None,
        path_provider: Optional[PathProvider] = None,
        test_mode: bool = False
    ):
        """Initialize utilities with configurable providers.
        
        Args:
            dataframe_provider: Provider for DataFrame operations
            file_stats_provider: Provider for file statistics
            path_provider: Provider for path operations
            test_mode: Enable test-specific behavior
        """
        self.dataframe_provider = dataframe_provider or DefaultDataFrameProvider()
        self.file_stats_provider = file_stats_provider or DefaultFileStatsProvider()
        self.path_provider = path_provider or DefaultPathProvider()
        self.test_mode = test_mode
        
        logger.debug(f"DataFrameUtilities initialized with test_mode={test_mode}")
    
    def _handle_list_files(
        self,
        files: List[str],
        include_stats: bool = False,
        base_directory: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """Handle list-based file input for DataFrame creation.
        
        Args:
            files: List of file paths
            include_stats: Whether to include file statistics
            base_directory: Optional base directory for relative paths
            
        Returns:
            DataFrame with file information
        """
        try:
            if not files:
                logger.info("Empty file list provided")
                return self.dataframe_provider.create_dataframe(columns=["path"])

            logger.debug(f"Processing {len(files)} files from list")
            df = self.dataframe_provider.create_dataframe({"path": files})

            # Add file stats if requested
            if include_stats:
                df = self._add_file_statistics(df, files)

            # Add relative paths if base_directory is specified
            if base_directory:
                df = self._add_relative_paths(df, base_directory)

            return df
            
        except Exception as e:
            logger.error(f"Failed to handle list files: {e}")
            raise ValueError(f"Error processing file list: {e}") from e
    
    def _handle_dict_files(
        self,
        files: Dict[str, Dict[str, Any]],
        include_stats: bool = False,
        base_directory: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """Handle dictionary-based file input for DataFrame creation.
        
        Args:
            files: Dictionary mapping file paths to metadata
            include_stats: Whether to include file statistics
            base_directory: Optional base directory for relative paths
            
        Returns:
            DataFrame with file information and metadata
        """
        try:
            if not files:
                logger.info("Empty file dictionary provided")
                return self.dataframe_provider.create_dataframe(columns=["path"])

            logger.debug(f"Processing {len(files)} files from dictionary")
            records = []
            
            for file_path, metadata in files.items():
                record = {"path": file_path, **metadata}

                # Add file stats if requested and not already present
                if include_stats and self._needs_file_stats(metadata):
                    record = self._merge_file_stats(record, file_path)
                
                # Add relative path if base_directory is specified
                if base_directory:
                    record = self._add_relative_path_to_record(record, file_path, base_directory)

                records.append(record)

            return self.dataframe_provider.create_dataframe(records)
            
        except Exception as e:
            logger.error(f"Failed to handle dict files: {e}")
            raise ValueError(f"Error processing file dictionary: {e}") from e
    
    def _add_file_statistics(self, df: pd.DataFrame, files: List[str]) -> pd.DataFrame:
        """Add file statistics to DataFrame.
        
        Args:
            df: Base DataFrame with file paths
            files: List of file paths to get stats for
            
        Returns:
            DataFrame with file statistics merged
        """
        try:
            stats_dfs = []
            successful_stats = 0
            
            for file_path in files:
                try:
                    if self.test_mode:
                        logger.debug(f"Test mode: Getting stats for {file_path}")
                    
                    stats = self.file_stats_provider.get_stats(file_path)
                    stats["path"] = file_path
                    stats_dfs.append(self.dataframe_provider.create_dataframe([stats]))
                    successful_stats += 1
                    
                except (FileNotFoundError, PermissionError) as e:
                    logger.warning(f"Skipping stats for {file_path}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error getting stats for {file_path}: {e}")
                    if self.test_mode:
                        raise  # Re-raise in test mode for debugging
                    continue

            if stats_dfs:
                logger.debug(f"Successfully collected stats for {successful_stats}/{len(files)} files")
                stats_df = self.dataframe_provider.concat_dataframes(stats_dfs, ignore_index=True)
                return self.dataframe_provider.merge_dataframes(df, stats_df, on="path", how="left")
            else:
                logger.warning("No file statistics could be collected")
                return df
                
        except Exception as e:
            logger.error(f"Failed to add file statistics: {e}")
            raise
    
    def _add_relative_paths(self, df: pd.DataFrame, base_directory: Union[str, Path]) -> pd.DataFrame:
        """Add relative paths column to DataFrame.
        
        Args:
            df: DataFrame with 'path' column
            base_directory: Base directory for relative path calculation
            
        Returns:
            DataFrame with 'relative_path' column added
        """
        try:
            logger.debug(f"Adding relative paths with base directory: {base_directory}")
            
            def get_relative_safe(path_str: str) -> str:
                """Safely get relative path with error handling."""
                try:
                    return str(self.path_provider.get_relative_path(path_str, base_directory))
                except Exception as e:
                    logger.warning(f"Failed to get relative path for {path_str}: {e}")
                    # Fallback to original path
                    return path_str
            
            df = df.copy()
            df["relative_path"] = df["path"].apply(get_relative_safe)
            return df
            
        except Exception as e:
            logger.error(f"Failed to add relative paths: {e}")
            raise
    
    def _needs_file_stats(self, metadata: Dict[str, Any]) -> bool:
        """Check if file statistics are needed for metadata.
        
        Args:
            metadata: Existing metadata dictionary
            
        Returns:
            True if file stats should be added
        """
        return all(
            key not in metadata for key in ["size", "mtime", "ctime"]
        )
    
    def _merge_file_stats(self, record: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """Merge file statistics into record.
        
        Args:
            record: Existing record dictionary
            file_path: Path to get statistics for
            
        Returns:
            Record with file statistics merged
        """
        try:
            with contextlib.suppress(FileNotFoundError, PermissionError):
                stats = self.file_stats_provider.get_stats(file_path)
                record.update(stats)
            return record
        except Exception as e:
            logger.warning(f"Failed to merge file stats for {file_path}: {e}")
            return record
    
    def _add_relative_path_to_record(
        self, 
        record: Dict[str, Any], 
        file_path: str, 
        base_directory: Union[str, Path]
    ) -> Dict[str, Any]:
        """Add relative path to record.
        
        Args:
            record: Record dictionary to update
            file_path: File path for relative calculation
            base_directory: Base directory
            
        Returns:
            Record with relative_path added
        """
        try:
            relative_path = self.path_provider.get_relative_path(file_path, base_directory)
            record["relative_path"] = str(relative_path)
        except Exception as e:
            logger.warning(f"Failed to add relative path for {file_path}: {e}")
            # Fallback to original path
            record["relative_path"] = file_path
        
        return record


def combine_metadata_and_data(data: Dict[str, Any], metadata: Dict[str, Any], 
                          prefix: str = "meta_", inplace: bool = False) -> Dict[str, Any]:
    """
    Combine experimental data with metadata into a single dictionary.
    
    This function merges metadata into the data dictionary with an optional prefix
    to avoid key collisions. It handles nested dictionaries and ensures that 
    metadata doesn't overwrite existing data keys.
    
    Args:
        data: Dictionary containing experimental data
        metadata: Dictionary containing metadata to be added
        prefix: Prefix to add to metadata keys to avoid collisions
        inplace: If True, modify the data dictionary in place
                
    Returns:
        Dictionary with combined data and metadata
        
    Raises:
        TypeError: If either data or metadata is not a dictionary
        ValueError: If there are key conflicts that can't be resolved with prefixing
    """
    if not isinstance(data, dict):
        raise TypeError(f"data must be a dictionary, got {type(data).__name__}")
    if not isinstance(metadata, dict):
        raise TypeError(f"metadata must be a dictionary, got {type(metadata).__name__}")
    
    # Create a copy if not modifying in place
    result = data if inplace else data.copy()
    
    for key, value in metadata.items():
        # Skip None values in metadata
        if value is None:
            continue
            
        # Handle nested dictionaries recursively
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = combine_metadata_and_data(
                result[key], 
                value, 
                prefix=prefix,
                inplace=False  # Always create new dict for nested updates
            )
        else:
            # Add prefix to avoid key collisions
            prefixed_key = f"{prefix}{key}" if prefix else key
            
            # Check for key conflicts
            if prefixed_key in result and result[prefixed_key] != value:
                raise ValueError(
                    f"Key conflict for '{prefixed_key}'. "
                    f"Existing value: {result[prefixed_key]}, New value: {value}"
                )
            
            # Add the metadata with prefix
            result[prefixed_key] = value
    
    return result


# Global instance for backward compatibility and standard usage
_default_utilities = DataFrameUtilities()


def attach_file_metadata_to_dataframe(
    df: pd.DataFrame,
    file_metadata: Dict[str, Dict[str, Any]],
    path_column: str = 'path',
    inplace: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Attach file metadata to an existing DataFrame.
    
    This function merges file metadata into an existing DataFrame based on
    matching file paths. The metadata is joined on the specified path column.
    
    Args:
        df: Input DataFrame containing file paths
        file_metadata: Dictionary mapping file paths to metadata dictionaries
        path_column: Name of the column containing file paths
        inplace: If True, modify the input DataFrame in place
        **kwargs: Additional arguments passed to pandas.merge
        
    Returns:
        DataFrame with attached metadata
        
    Raises:
        ValueError: If path_column is not in the DataFrame
        KeyError: If path_column is not found in the DataFrame
        
    Example:
        >>> df = pd.DataFrame({
        ...     'path': ['/data/file1.txt', '/data/file2.txt'],
        ...     'value': [1, 2]
        ... })
        >>> metadata = {
        ...     '/data/file1.txt': {'animal': 'mouse', 'condition': 'test'},
        ...     '/data/file2.txt': {'animal': 'rat', 'condition': 'control'}
        ... }
        >>> result = attach_file_metadata_to_dataframe(df, metadata)
    """
    if not inplace:
        df = df.copy()
    
    if path_column not in df.columns:
        raise KeyError(f"Column '{path_column}' not found in DataFrame columns: {df.columns.tolist()}")
    
    # Convert metadata to DataFrame
    metadata_df = pd.DataFrame.from_dict(file_metadata, orient='index')
    metadata_df.index.name = path_column
    metadata_df = metadata_df.reset_index()
    
    # Merge with original DataFrame
    merged_df = df.merge(metadata_df, on=path_column, how='left', **kwargs)
    
    return merged_df


def discovery_results_to_dataframe(
    discovery_results: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]],
    include_stats: bool = False,
    base_directory: Optional[Union[str, Path]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Convert discovery results to a pandas DataFrame.
    
    This is a high-level convenience function that wraps build_manifest_df
    with a more user-friendly interface for common discovery result formats.
    
    Args:
        discovery_results: Discovery results to convert. Can be:
            - A list of file paths (strings)
            - A list of dictionaries with file metadata
            - A dictionary mapping file paths to metadata dictionaries
        include_stats: Whether to include file statistics (size, mtime, etc.)
        base_directory: Optional base directory for relative paths
        **kwargs: Additional arguments passed to build_manifest_df
        
    Returns:
        DataFrame with file information and metadata
        
    Raises:
        ValueError: If input data is invalid or processing fails
        TypeError: If discovery_results is not a list or dict
        
    Example:
        >>> results = [
        ...     {"path": "/data/file1.txt", "animal": "mouse", "condition": "test"},
        ...     {"path": "/data/file2.txt", "animal": "rat", "condition": "control"},
        ... ]
        >>> df = discovery_results_to_dataframe(results)
    """
    # Handle different input formats
    if isinstance(discovery_results, list):
        if not discovery_results:
            return pd.DataFrame()
            
        # Convert list of dicts to dict format if needed
        if all(isinstance(item, dict) for item in discovery_results):
            # If all items have a 'path' key, use that as the dict key
            if all('path' in item for item in discovery_results):
                discovery_results = {
                    item['path']: {k: v for k, v in item.items() if k != 'path'}
                    for item in discovery_results
                }
            else:
                # Otherwise, convert list to dict with sequential keys
                discovery_results = {str(i): item for i, item in enumerate(discovery_results)}
    
    # At this point, discovery_results should be a dict
    if not isinstance(discovery_results, dict):
        raise TypeError(
            f"discovery_results must be a list or dict, got {type(discovery_results).__name__}"
        )
    
    # Use build_manifest_df for the actual conversion
    return build_manifest_df(
        files=discovery_results,
        include_stats=include_stats,
        base_directory=base_directory,
        **kwargs
    )


def build_manifest_df(
    files: Union[List[str], Dict[str, Dict[str, Any]]],
    include_stats: bool = False,
    base_directory: Optional[Union[str, Path]] = None,
    # Test-specific parameters for dependency injection
    dataframe_provider: Optional[DataFrameProvider] = None,
    file_stats_provider: Optional[FileStatsProvider] = None,
    path_provider: Optional[PathProvider] = None,
    test_mode: bool = False
) -> pd.DataFrame:
    """
    Convert discovery results to a pandas DataFrame.
    
    Enhanced with dependency injection patterns for comprehensive testing.
    
    Args:
        files: Discovery results (either list of paths or dict with metadata)
        include_stats: Whether to include file statistics (size, mtime, ctime, creation_time)
        base_directory: Optional base directory for calculating relative paths
        dataframe_provider: Optional DataFrame provider for testing (TST-REF-001)
        file_stats_provider: Optional file stats provider for testing (F-016)
        path_provider: Optional path provider for testing (F-016)
        test_mode: Enable test-specific behavior for debugging (TST-REF-003)
        
    Returns:
        DataFrame with file information and metadata
        
    Raises:
        ValueError: If input data is invalid or processing fails
        TypeError: If files parameter is not list or dict
    """
    logger.debug(f"Building manifest DataFrame for {type(files)} with {len(files) if files else 0} items")
    
    try:
        # Use injected providers if provided (for testing), otherwise use defaults
        if any([dataframe_provider, file_stats_provider, path_provider, test_mode]):
            utilities = DataFrameUtilities(
                dataframe_provider=dataframe_provider,
                file_stats_provider=file_stats_provider,
                path_provider=path_provider,
                test_mode=test_mode
            )
        else:
            utilities = _default_utilities
        
        # Handle list of files
        if isinstance(files, list):
            logger.debug("Processing list-based file input")
            return utilities._handle_list_files(files, include_stats, base_directory)
        
        # Handle dictionary with metadata
        elif isinstance(files, dict):
            logger.debug("Processing dictionary-based file input")
            return utilities._handle_dict_files(files, include_stats, base_directory)
        
        else:
            logger.error(f"Invalid files type: {type(files)}")
            raise TypeError(f"Files must be list or dict, got {type(files)}")
            
    except Exception as e:
        logger.error(f"Failed to build manifest DataFrame: {e}")
        raise


def filter_manifest_df(
    df: pd.DataFrame,
    test_mode: bool = False,
    **filters: Any
) -> pd.DataFrame:
    """
    Filter a manifest DataFrame based on column values.
    
    Enhanced with improved error handling and test-specific entry points.
    
    Args:
        df: DataFrame to filter
        test_mode: Enable test-specific behavior and detailed logging (TST-REF-003)
        **filters: Column-value pairs for filtering (e.g., animal='mouse', condition='test')
        
    Returns:
        Filtered DataFrame
        
    Raises:
        ValueError: If DataFrame is invalid or filtering fails
    """
    logger.debug(f"Filtering DataFrame with shape {df.shape} using {len(filters)} filters")
    
    if test_mode:
        logger.debug(f"Test mode: Filter criteria: {filters}")
    
    try:
        # Validate input DataFrame
        if df.empty:
            logger.warning("Empty DataFrame provided for filtering")
            return df.copy()
        
        filtered_df = df.copy()
        applied_filters = 0
        
        for column, value in filters.items():
            if column in filtered_df.columns:
                original_count = len(filtered_df)
                
                if isinstance(value, list):
                    logger.debug(f"Applying list filter: {column} in {value}")
                    filtered_df = filtered_df[filtered_df[column].isin(value)]
                else:
                    logger.debug(f"Applying equality filter: {column} == {value}")
                    filtered_df = filtered_df[filtered_df[column] == value]
                
                filtered_count = len(filtered_df)
                applied_filters += 1
                
                if test_mode:
                    logger.debug(f"Filter {column}: {original_count} -> {filtered_count} rows")
            else:
                logger.warning(f"Column '{column}' not found in DataFrame columns: {list(df.columns)}")
                if test_mode:
                    raise ValueError(f"Column '{column}' not found in DataFrame")
        
        logger.debug(f"Applied {applied_filters} filters, final DataFrame shape: {filtered_df.shape}")
        return filtered_df
        
    except Exception as e:
        logger.error(f"Failed to filter DataFrame: {e}")
        if test_mode:
            raise  # Re-raise in test mode for debugging
        # In production mode, return original DataFrame on filter failure
        logger.warning("Returning original DataFrame due to filter failure")
        return df.copy()


def extract_unique_values(
    df: pd.DataFrame,
    column: str,
    test_mode: bool = False,
    handle_missing: str = "drop"
) -> List[Any]:
    """
    Extract unique values from a column in a manifest DataFrame.
    
    Enhanced with improved error handling and configurable missing value handling.
    
    Args:
        df: DataFrame to extract from
        column: Column name to get unique values from
        test_mode: Enable test-specific behavior and detailed logging (TST-REF-003)
        handle_missing: How to handle missing values - "drop", "keep", or "error"
        
    Returns:
        List of unique values in the column
        
    Raises:
        ValueError: If column not found or handle_missing is invalid
        KeyError: If column doesn't exist and test_mode is True
    """
    logger.debug(f"Extracting unique values from column '{column}' in DataFrame with shape {df.shape}")
    
    if test_mode:
        logger.debug(f"Test mode: handle_missing={handle_missing}")
    
    try:
        # Validate handle_missing parameter
        if handle_missing not in ["drop", "keep", "error"]:
            raise ValueError(f"handle_missing must be 'drop', 'keep', or 'error', got '{handle_missing}'")
        
        # Check if column exists
        if column not in df.columns:
            logger.warning(f"Column '{column}' not found in DataFrame columns: {list(df.columns)}")
            if test_mode:
                raise KeyError(f"Column '{column}' not found in DataFrame")
            return []
        
        # Handle empty DataFrame
        if df.empty:
            logger.warning("Empty DataFrame provided for unique value extraction")
            return []
        
        column_data = df[column]
        original_count = len(column_data)
        missing_count = column_data.isna().sum()
        
        if test_mode and missing_count > 0:
            logger.debug(f"Found {missing_count} missing values out of {original_count} total")
        
        # Handle missing values based on strategy
        if handle_missing == "drop":
            processed_data = column_data.dropna()
        elif handle_missing == "keep":
            processed_data = column_data
        elif handle_missing == "error" and missing_count > 0:
            raise ValueError(f"Found {missing_count} missing values in column '{column}'")
        else:
            processed_data = column_data
        
        unique_values = processed_data.unique().tolist()
        
        logger.debug(f"Extracted {len(unique_values)} unique values from column '{column}'")
        
        if test_mode:
            logger.debug(f"Test mode: unique values preview: {unique_values[:5]}...")
        
        return unique_values
        
    except Exception as e:
        logger.error(f"Failed to extract unique values from column '{column}': {e}")
        if test_mode:
            raise  # Re-raise in test mode for debugging
        # In production mode, return empty list on failure
        logger.warning("Returning empty list due to extraction failure")
        return []