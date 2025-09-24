"""Data loading and transformation public API entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from flyrigloader import logger
from flyrigloader.config.models import LegacyConfigAdapter
from flyrigloader.exceptions import FlyRigLoaderError
from flyrigloader.io.column_models import get_config_from_source as _get_config_from_source
from flyrigloader.io.loaders import load_data_file as _load_data_file
from flyrigloader.io.transformers import transform_to_dataframe as _transform_to_dataframe

from .configuration import (
    load_and_validate_config as _load_and_validate_config,
    resolve_config_source as _resolve_config_source,
)
from .dependencies import DefaultDependencyProvider, get_dependency_provider
from .helpers import _attach_metadata_bucket
from .paths import _resolve_base_directory

def load_data_file(
    file_path: Union[str, Path],
    validate_format: bool = True,
    loader: Optional[str] = None,
    _deps: Optional[DefaultDependencyProvider] = None
) -> Dict[str, Any]:
    """
    Load raw data from a single file without DataFrame transformation.
    
    This function implements the second step of the new decoupled architecture,
    providing selective data loading from individual files using the registry-based
    loader system. This enables memory-efficient processing of large datasets and
    selective analysis workflows.
    
    Args:
        file_path: Path to the data file to load
        validate_format: Whether to validate the loaded data format (default True)
        loader: Optional loader identifier for explicit loader selection
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        Dictionary containing raw experimental data from the file
        
    Raises:
        FileNotFoundError: If the data file doesn't exist
        ValueError: If the file format is invalid or data cannot be loaded
        
    Example:
        >>> # Load individual files from manifest
        >>> manifest = discover_experiment_manifest(...)
        >>> for file_path in manifest.keys():
        ...     raw_data = load_data_file(file_path)
        ...     print(f"Loaded {len(raw_data)} data columns from {file_path}")
    """
    operation_name = "load_data_file"
    
    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()
    
    logger.debug(f"📁 Loading data file: {file_path}")
    
    # Validate file_path parameter
    if not file_path:
        error_msg = (
            f"Invalid file_path for {operation_name}: '{file_path}'. "
            "file_path must be a non-empty string or Path object pointing to the data file."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    # Use the new decoupled loading function
    try:
        logger.debug(f"Loading raw data using decoupled loader: {file_path}")
        raw_data = _load_data_file(file_path, loader)
        
        # Validate format if requested
        if validate_format:
            if not isinstance(raw_data, dict):
                logger.warning(f"Expected dictionary data structure, got {type(raw_data).__name__}")
                if validate_format:
                    raise ValueError(f"Invalid data format: expected dict, got {type(raw_data).__name__}")
            else:
                data_keys = list(raw_data.keys())
                logger.debug(f"Loaded data with {len(data_keys)} columns: {data_keys}")
                
                # Basic validation for common required keys
                if 't' not in raw_data:
                    logger.warning("Time column 't' not found in data - may cause issues in downstream processing")
        
        file_path_obj = Path(file_path)
        file_size = file_path_obj.stat().st_size if file_path_obj.exists() else 0
        logger.debug(f"✓ Successfully loaded {len(raw_data) if isinstance(raw_data, dict) else 'N/A'} data columns from {file_path}")
        logger.debug(f"  File size: {file_size:,} bytes")
        
        return raw_data
        
    except Exception as e:
        error_msg = (
            f"Failed to load data from {file_path} for {operation_name}: {e}. "
            "Please check the file format and ensure it's a valid data file."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from e

def transform_to_dataframe(
    raw_data: Dict[str, Any],
    column_config_path: Optional[Union[str, Path, Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    add_file_path: bool = True,
    file_path: Optional[Union[str, Path]] = None,
    strict_schema: bool = False,
    _deps: Optional[DefaultDependencyProvider] = None
) -> pd.DataFrame:
    """
    Transform raw experimental data into a pandas DataFrame with optional configuration.
    
    This function implements the third step of the new decoupled architecture,
    providing optional DataFrame transformation from raw data. This enables
    selective processing and memory-efficient workflows where not all data
    needs to be converted to DataFrames.
    
    Args:
        raw_data: Dictionary containing raw experimental data
        column_config_path: Path to column configuration file, configuration dictionary,
                           or ColumnConfigDict instance. If None, uses default configuration.
        metadata: Optional dictionary of metadata to add to the DataFrame
        add_file_path: Whether to add a 'file_path' column with source file path (default True)
        file_path: Source file path to add to 'file_path' column (required if add_file_path=True)
        strict_schema: If True, drop any columns not present in the column configuration
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        pd.DataFrame: Processed experimental data with configured columns and metadata
        
    Raises:
        ValueError: If required columns are missing from the data or configuration is invalid
        TypeError: If input types are invalid
        
    Example:
        >>> # Transform only selected files
        >>> manifest = discover_experiment_manifest(...)
        >>> dataframes = []
        >>> for file_path in list(manifest.keys())[:5]:  # Process only first 5 files
        ...     raw_data = load_data_file(file_path)
        ...     df = transform_to_dataframe(raw_data, file_path=file_path)
        ...     dataframes.append(df)
        >>> combined_df = pd.concat(dataframes, ignore_index=True)
    """
    operation_name = "transform_to_dataframe"
    
    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()
    
    logger.debug(f"🔄 Transforming raw data to DataFrame")
    
    # Validate raw_data parameter
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
    
    # Validate file_path requirement when add_file_path=True
    if add_file_path and not file_path:
        error_msg = (
            f"file_path parameter required when add_file_path=True for {operation_name}. "
            "Please provide the source file path or set add_file_path=False."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    # Log transformation details
    data_columns = list(raw_data.keys())
    logger.debug(f"Transforming {len(data_columns)} data columns: {data_columns}")
    if metadata:
        logger.debug(f"Adding metadata: {list(metadata.keys())}")
    
    # Use the new decoupled transformation function
    try:
        logger.debug("Calling DataFrame transformation utility from decoupled architecture")
        df = _transform_to_dataframe(
            exp_matrix=raw_data,
            config_source=column_config_path,
            metadata=metadata
        )
        
        # Add file path column if requested
        if add_file_path and file_path:
            file_path_obj = Path(file_path)
            df["file_path"] = str(file_path_obj.resolve())
            logger.debug(f"Added file_path column: {file_path}")
        
        # Apply strict schema filtering if requested
        if strict_schema:
            if column_config_path is None:
                raise FlyRigLoaderError(
                    "strict_schema=True requires a column_config_path (schema) to be provided"
                )
            try:
                schema_model = _get_config_from_source(column_config_path)
                allowed_cols = set(schema_model.columns.keys())
                if add_file_path:
                    allowed_cols.add("file_path")  # Always allow file_path column
                
                if extra_cols := [c for c in df.columns if c not in allowed_cols]:
                    logger.debug(f"Dropping {len(extra_cols)} columns not in schema: {extra_cols}")
                    df = df[list(allowed_cols & set(df.columns))]
            except Exception as e:
                raise FlyRigLoaderError(
                    f"Failed to load column configuration for strict schema filtering: {e}"
                ) from e
        
        logger.debug(f"✓ Successfully transformed to DataFrame with shape: {df.shape}")
        logger.debug(f"  DataFrame columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        error_msg = (
            f"Failed to transform raw data to DataFrame for {operation_name}: {e}. "
            "Please check the data structure and column configuration compatibility."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from e

def load_experiment_files(
    config: Optional[Union[Dict[str, Any], Any]] = None,
    experiment_name: str = "",
    config_path: Optional[Union[str, Path]] = None,
    base_directory: Optional[Union[str, Path]] = None,
    pattern: str = "*.*",
    recursive: bool = True,
    extensions: Optional[List[str]] = None,
    extract_metadata: bool = False,
    parse_dates: bool = False,
    _deps: Optional[DefaultDependencyProvider] = None
) -> Union[List[str], Dict[str, Dict[str, Any]]]:
    """
    High-level function to load files for a specific experiment with enhanced testability.
    
    This function supports comprehensive dependency injection for testing scenarios
    through the _deps parameter, enabling pytest.monkeypatch patterns. It now accepts
    Pydantic models directly for improved type safety.
    
    Args:
        config: Pre-loaded configuration dictionary, Pydantic model, or LegacyConfigAdapter
        experiment_name: Name of the experiment to load files for
        config_path: Path to the YAML configuration file (alternative to config)
        base_directory: Optional override for the data directory (if not specified, uses config)
        pattern: File pattern to search for in glob format (e.g., "*.csv", "data_*.pkl")
        recursive: Whether to search recursively (defaults to True)
        extensions: Optional list of file extensions to filter by
        extract_metadata: If True, extract metadata from filenames using config patterns
        parse_dates: If True, attempt to parse dates from filenames
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        If extract_metadata or parse_dates is True: Dictionary mapping file paths to metadata.
        Each entry contains the flattened metadata fields and a ``metadata`` bucket with the
        same information for convenient downstream access. Otherwise: List of file paths for
        the experiment.
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        KeyError: If the experiment doesn't exist in the config
        ValueError: If neither config_path nor config is provided, or if both are provided
        
    Example:
        >>> # Using Pydantic model directly
        >>> config = create_config(
        ...     project_name="fly_behavior",
        ...     base_directory="/data/experiments"
        ... )
        >>> files = load_experiment_files(
        ...     config=config,
        ...     experiment_name="plume_navigation"
        ... )
        >>> print(f"Found {len(files)} files")
    """
    operation_name = "load_experiment_files"
    
    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()
    
    logger.info(f"Loading experiment files for experiment '{experiment_name}'")
    logger.debug(f"Parameters: pattern={pattern}, recursive={recursive}, "
                f"extensions={extensions}, extract_metadata={extract_metadata}, "
                f"parse_dates={parse_dates}")
    
    # Validate experiment_name parameter
    if not experiment_name or not isinstance(experiment_name, str):
        error_msg = (
            f"Invalid experiment_name for {operation_name}: '{experiment_name}'. "
            "experiment_name must be a non-empty string representing the experiment identifier."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    config_dict = _resolve_config_source(config, config_path, operation_name, _deps)
    
    # Determine the data directory with enhanced validation
    base_directory = _resolve_base_directory(config_dict, base_directory, operation_name)
    
    logger.debug(f"Using base directory: {base_directory}")
    
    # Validate experiment exists in configuration
    try:
        experiment_info = _deps.config.get_experiment_info(config_dict, experiment_name)
        logger.debug(f"Found experiment configuration for '{experiment_name}'")
    except KeyError as e:
        available_experiments = list(config_dict.get("experiments", {}).keys())
        error_msg = (
            f"Experiment '{experiment_name}' not found in configuration. "
            f"Available experiments: {available_experiments}. "
            "Please check the experiment name and ensure it's defined in your configuration."
        )
        logger.error(error_msg)
        raise KeyError(error_msg) from e
    
    # Discover experiment files with dependency injection
    try:
        logger.debug(f"Discovering files for experiment '{experiment_name}'")
        result = _deps.discovery.discover_experiment_files(
            config=config_dict,
            experiment_name=experiment_name,
            base_directory=base_directory,
            pattern=pattern,
            recursive=recursive,
            extensions=extensions,
            extract_metadata=extract_metadata,
            parse_dates=parse_dates
        )

        if extract_metadata or parse_dates:
            result = _attach_metadata_bucket(result)

        file_count = len(result) if isinstance(result, (list, dict)) else 0
        logger.info(f"Successfully discovered {file_count} files for experiment '{experiment_name}'")
        return result

    except Exception as e:
        error_msg = (
            f"Failed to discover files for experiment '{experiment_name}': {e}. "
            "Please check the experiment configuration and data directory structure."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

def load_dataset_files(
    config: Optional[Union[Dict[str, Any], Any]] = None,
    dataset_name: str = "",
    config_path: Optional[Union[str, Path]] = None,
    base_directory: Optional[Union[str, Path]] = None,
    pattern: str = "*.*",
    recursive: bool = True,
    extensions: Optional[List[str]] = None,
    extract_metadata: bool = False,
    parse_dates: bool = False,
    _deps: Optional[DefaultDependencyProvider] = None
) -> Union[List[str], Dict[str, Dict[str, Any]]]:
    """
    High-level function to load files for a specific dataset with enhanced testability.
    
    This function supports comprehensive dependency injection for testing scenarios
    through the _deps parameter, enabling pytest.monkeypatch patterns. It now accepts
    Pydantic models directly for improved type safety.
    
    Args:
        config: Pre-loaded configuration dictionary, Pydantic model, or LegacyConfigAdapter
        dataset_name: Name of the dataset to load files for
        config_path: Path to the YAML configuration file (alternative to config)
        base_directory: Optional override for the data directory (if not specified, uses config)
        pattern: File pattern to search for in glob format (e.g., "*.csv", "data_*.pkl")
        recursive: Whether to search recursively (defaults to True)
        extensions: Optional list of file extensions to filter by
        extract_metadata: If True, extract metadata from filenames using config patterns
        parse_dates: If True, attempt to parse dates from filenames
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        If extract_metadata or parse_dates is True: Dictionary mapping file paths to metadata.
        Each entry contains the flattened metadata fields and a ``metadata`` bucket with the
        same information for convenient downstream access. Otherwise: List of file paths for
        the dataset.
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        KeyError: If the dataset doesn't exist in the config
        ValueError: If neither config_path nor config is provided, or if both are provided
        
    Example:
        >>> # Using Pydantic model directly
        >>> config = create_config(
        ...     project_name="fly_behavior",
        ...     base_directory="/data/experiments"
        ... )
        >>> files = load_dataset_files(
        ...     config=config,
        ...     dataset_name="plume_tracking"
        ... )
        >>> print(f"Found {len(files)} files")
    """
    operation_name = "load_dataset_files"
    
    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()
    
    logger.info(f"Loading dataset files for dataset '{dataset_name}'")
    logger.debug(f"Parameters: pattern={pattern}, recursive={recursive}, "
                f"extensions={extensions}, extract_metadata={extract_metadata}, "
                f"parse_dates={parse_dates}")
    
    # Validate dataset_name parameter
    if not dataset_name or not isinstance(dataset_name, str):
        error_msg = (
            f"Invalid dataset_name for {operation_name}: '{dataset_name}'. "
            "dataset_name must be a non-empty string representing the dataset identifier."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    config_dict = _resolve_config_source(config, config_path, operation_name, _deps)
    
    # Determine the data directory with enhanced validation
    base_directory = _resolve_base_directory(config_dict, base_directory, operation_name)
    
    logger.debug(f"Using base directory: {base_directory}")
    
    # Validate dataset exists in configuration
    try:
        dataset_info = _deps.config.get_dataset_info(config_dict, dataset_name)
        logger.debug(f"Found dataset configuration for '{dataset_name}'")
    except KeyError as e:
        available_datasets = list(config_dict.get("datasets", {}).keys())
        error_msg = (
            f"Dataset '{dataset_name}' not found in configuration. "
            f"Available datasets: {available_datasets}. "
            "Please check the dataset name and ensure it's defined in your configuration."
        )
        logger.error(error_msg)
        raise KeyError(error_msg) from e
    
    # Discover dataset files with dependency injection
    try:
        logger.debug(f"Discovering files for dataset '{dataset_name}'")
        result = _deps.discovery.discover_dataset_files(
            config=config_dict,
            dataset_name=dataset_name,
            base_directory=base_directory,
            pattern=pattern,
            recursive=recursive,
            extensions=extensions,
            extract_metadata=extract_metadata,
            parse_dates=parse_dates
        )

        if extract_metadata or parse_dates:
            result = _attach_metadata_bucket(result)

        file_count = len(result) if isinstance(result, (list, dict)) else 0
        logger.info(f"Successfully discovered {file_count} files for dataset '{dataset_name}'")
        return result
        
    except Exception as e:
        error_msg = (
            f"Failed to discover files for dataset '{dataset_name}': {e}. "
            "Please check the dataset configuration and data directory structure."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

def process_experiment_data(
    data_path: Union[str, Path],
    *,
    column_config_path: Optional[Union[str, Path, Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    strict_schema: bool = False,
    _deps: Optional[DefaultDependencyProvider] = None
) -> pd.DataFrame:
    """
    Process experimental data and return a pandas DataFrame with a guaranteed `file_path` column.
    
    .. deprecated:: 
        This function is deprecated. The monolithic approach is less flexible. 
        Use the new decoupled architecture with load_data_file() + transform_to_dataframe() instead.
    
    This function maintains backward compatibility while internally using the new decoupled
    architecture (load_data_file + transform_to_dataframe). It provides the same API surface
    as before but with enhanced logging and validation.
    
    Args:
        data_path: Path to the pickle file containing experimental data
        column_config_path: Path to column configuration file, configuration dictionary,
                            or ColumnConfigDict instance. If None, uses default configuration.
        metadata: Optional dictionary of metadata to add to the DataFrame
        strict_schema: If True, drop any columns not present in the provided column
            configuration. Requires that ``column_config_path`` is supplied. This
            option is useful for downstream pipelines that rely on a strict
            schema definition (e.g. Kedro parameters).
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        pd.DataFrame: Processed experimental data. Always contains a `file_path` column with the absolute path to the source pickle.
        
    Raises:
        FileNotFoundError: If the data or config file doesn't exist
        ValueError: If required columns are missing from the data or path is invalid
        
    Example:
        >>> # Deprecated approach
        >>> df = process_experiment_data("experiment_data.pkl")
        >>> print(f"Processed {len(df)} rows with columns: {list(df.columns)}")
        
        >>> # New decoupled approach (recommended)
        >>> raw_data = load_data_file("experiment_data.pkl")
        >>> df = transform_to_dataframe(raw_data, file_path="experiment_data.pkl")
        >>> print(f"Processed {len(df)} rows with columns: {list(df.columns)}")
    """
    operation_name = "process_experiment_data"

    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()

    logger.info(f"📊 Processing experimental data from: {data_path}")
    logger.debug(f"Using backward-compatible API with new decoupled architecture")

    # Use the new decoupled architecture internally
    try:
        # Step 1: Load raw data (replaces direct pickle loading)
        logger.debug("Step 1: Loading raw data using decoupled architecture")
        raw_data = load_data_file(
            file_path=data_path,
            validate_format=True,
            _deps=_deps
        )
        
        # Step 2: Transform to DataFrame (replaces make_dataframe_from_config)
        logger.debug("Step 2: Transforming to DataFrame using decoupled architecture")
        df = transform_to_dataframe(
            raw_data=raw_data,
            column_config_path=column_config_path,
            metadata=metadata,
            add_file_path=True,
            file_path=data_path,
            strict_schema=strict_schema,
            _deps=_deps
        )
        
        logger.info(f"✓ Successfully processed experimental data")
        logger.info(f"  DataFrame shape: {df.shape}")
        logger.info(f"  Source file: {data_path}")
        logger.debug(f"  DataFrame columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        # Re-raise with original API error format for backward compatibility
        error_msg = (
            f"Failed to process experimental data from {data_path}: {e}. "
            "Please check the file format and column configuration compatibility."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from e
