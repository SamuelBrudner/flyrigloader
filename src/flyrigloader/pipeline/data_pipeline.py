"""
data_pipeline.py - Complete loading pipeline for fly rig data.

This module provides high-level functions that orchestrate the entire process
of loading fly rig data, from configuration loading to returning a merged DataFrame.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
from pathlib import Path
from loguru import logger
from datetime import datetime

from ..config_utils.manager import ConfigManager, get_config, create_config_manager
from ..config_utils.filter import filter_config_by_experiment
from ..discovery.files import discover_dataset_files
from ..schema.validator import get_schema_by_name
from ..lineage.tracker import LineageTracker  # Keep for type annotations
from ..core.utils import ensure_path, ensure_path_exists, PathLike
from .data_assembly import load_file_into_dataframe, combine_dataframes
from ..core.factories import create_lineage_tracker


def _create_metadata(success: bool = False, **kwargs) -> Dict[str, Any]:
    """
    Create a standardized metadata dictionary following the project's conventions.
    
    Args:
        success: Whether the operation was successful
        **kwargs: Additional metadata fields to include
    
    Returns:
        Metadata dictionary with standard fields
    """
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "success": success
    }
    metadata.update(kwargs)
    return metadata


def _create_error_metadata(error: Exception) -> Dict[str, Any]:
    """
    Create a standardized error metadata dictionary from an exception.
    
    Args:
        error: The exception to convert
        
    Returns:
        Dictionary with standardized error information
    """
    return _create_metadata(
        success=False,
        error=str(error),
        error_type=type(error).__name__
    )


def load_experiment_data(
    config: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
    *,
    track_lineage: bool = True,
    lineage_export_path: Optional[PathLike] = None,
    experiment_name: Optional[str] = None,
    dataset_names: Optional[List[str]] = None
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Load all data for an experiment based on configuration.
    
    This is the primary entry point for loading fly rig data. It orchestrates
    the entire process from configuration to returning a processed DataFrame.
    
    Args:
        config: Configuration dictionary from ConfigManager.get_config()
        schema: Optional schema for validation (uses config schema if None)
        track_lineage: Whether to track data lineage
        lineage_export_path: Optional path to export lineage information
        experiment_name: Optional name to filter for a specific experiment 
        dataset_names: Optional list to filter for specific datasets
        
    Returns:
        Tuple containing (DataFrame, metadata_dict) with the loaded data and processing info.
        If processing fails, DataFrame will be None and metadata_dict will contain error details.
    """
    try:
        # Validate and prepare the configuration
        config_result, config_meta = _prepare_experiment_config(config, experiment_name)
        if not config_meta["success"]:
            return None, config_meta
        
        prepared_config = config_result
        
        # Extract experiment and validate datasets
        dataset_result, dataset_meta = _validate_and_get_datasets(prepared_config, dataset_names)
        if not dataset_meta["success"]:
            return None, dataset_meta
        
        experiment, validated_datasets = dataset_result
        
        # Get schema if needed
        schema_result, schema_meta = _resolve_schema(experiment, schema, prepared_config)
        if not schema_meta["success"]:
            return None, schema_meta
        
        validated_schema = schema_result
        
        # Setup lineage tracking
        lineage_result, lineage_meta = _setup_lineage_tracking(track_lineage, experiment, validated_datasets)
        lineage = lineage_result if lineage_meta["success"] else None
        
        # Discover files
        file_items_result, file_items_meta = _discover_and_validate_files(
            validated_datasets, prepared_config, lineage
        )
        if not file_items_meta["success"]:
            return _empty_result(lineage, track_lineage, lineage_export_path), file_items_meta
        
        file_items = file_items_result
        
        # Load and process files
        data_result, data_meta = _load_and_process_files(
            file_items, validated_schema, lineage, track_lineage
        )
        if not data_meta["success"]:
            return _empty_result(lineage, track_lineage, lineage_export_path), data_meta
        
        result_df = data_result
        
        # Export lineage if requested
        if lineage and lineage_export_path:
            lineage_meta = _export_lineage(lineage, lineage_export_path)
            if not lineage_meta["success"]:
                logger.warning(lineage_meta.get("error", "Failed to export lineage"))
        
        logger.info(f"Loaded {len(result_df)} rows of data")
        return result_df, {"success": True, "rows_loaded": len(result_df)}
    except Exception as e:
        logger.error(f"Failed to load experiment data: {str(e)}")
        return None, _create_error_metadata(e)


def _prepare_experiment_config(
    config: Dict[str, Any], 
    experiment_name: Optional[str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Filter and prepare the configuration for a specific experiment.
    
    Args:
        config: The full configuration dictionary
        experiment_name: Optional name of experiment to filter for
        
    Returns:
        Tuple of (filtered_config, metadata)
    """
    try:
        metadata = _create_metadata()
        
        # Input validation
        if not isinstance(config, dict):
            metadata["error"] = f"Config must be a dictionary, got {type(config).__name__}"
            return {}, metadata
        
        # Filter for experiment if specified
        if experiment_name is not None:
            filter_result, filter_meta = filter_config_by_experiment(config, experiment_name)
            
            if not filter_meta["success"]:
                metadata.update({k: v for k, v in filter_meta.items() if k != "timestamp"})
                return {}, metadata
            
            return filter_result, {"timestamp": datetime.now().isoformat(), "success": True}
        
        return config, {"timestamp": datetime.now().isoformat(), "success": True}
    except Exception as e:
        logger.error(f"Failed to prepare experiment config: {str(e)}")
        return {}, _create_error_metadata(e)


def _validate_and_get_datasets(
    config: Dict[str, Any],
    dataset_names: Optional[List[str]]
) -> Tuple[Tuple[Dict[str, Any], List[str]], Dict[str, Any]]:
    """
    Extract experiment data and validate datasets.
    
    Args:
        config: The configuration dictionary
        dataset_names: Optional list of dataset names to use
        
    Returns:
        Tuple of ((experiment, datasets), metadata)
    """
    try:
        metadata = _create_metadata()
        
        # Extract experiment from config
        experiment = config.get("experiment", {})
        
        if not isinstance(experiment, dict):
            metadata["error"] = f"Expected experiment to be a dictionary, got {type(experiment).__name__}"
            return (({}, []), metadata)
        
        # Determine datasets to load
        validated_datasets = dataset_names if dataset_names is not None else experiment.get("datasets", [])
        
        if not isinstance(validated_datasets, list):
            metadata["error"] = f"Datasets must be a list, got {type(validated_datasets).__name__}"
            return ((experiment, []), metadata)
            
        if not validated_datasets:
            logger.warning("No datasets specified for loading")
            metadata["success"] = True
            metadata["warning"] = "No datasets specified"
            return ((experiment, []), metadata)
        
        metadata["success"] = True
        return ((experiment, validated_datasets), metadata)
    except Exception as e:
        logger.error(f"Failed to validate and get datasets: {str(e)}")
        return (({}, []), _create_error_metadata(e))


def _resolve_schema(
    experiment: Dict[str, Any],
    schema: Optional[Dict[str, Any]],
    config: Dict[str, Any]
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Resolve the schema to use for an experiment.
    
    Args:
        experiment: Experiment configuration
        schema: Explicit schema if provided
        config: The full configuration
        
    Returns:
        Tuple of (schema, metadata)
    """
    try:
        metadata = _create_metadata()
        
        # If schema is provided directly, validate and use it
        if schema is not None:
            if not isinstance(schema, dict):
                metadata["error"] = f"Schema must be a dictionary, got {type(schema).__name__}"
                return None, metadata
            
            metadata["success"] = True
            return schema, metadata
        
        # Try to get schema from experiment config
        schema_name = experiment.get("schema")
        if not schema_name:
            # No schema specified, which is valid
            metadata["success"] = True
            metadata["note"] = "No schema specified in experiment config"
            return None, metadata
        
        # Get schema by name
        schema_result, schema_meta = get_schema_by_name(schema_name, config=config)
        
        # Merge metadata but keep original success flag
        metadata.update({k: v for k, v in schema_meta.items() if k != "timestamp"})
        
        return schema_result, metadata
    except Exception as e:
        logger.error(f"Failed to resolve schema: {str(e)}")
        return None, _create_error_metadata(e)


def _setup_lineage_tracking(
    track_lineage: bool,
    experiment: Dict[str, Any],
    dataset_names: List[str]
) -> Tuple[Optional[LineageTracker], Dict[str, Any]]:
    """
    Set up lineage tracking if requested.
    
    Args:
        track_lineage: Whether to track lineage
        experiment: The experiment configuration
        dataset_names: List of dataset names
        
    Returns:
        Tuple of (lineage_tracker, metadata)
    """
    try:
        metadata = _create_metadata()
        
        if not track_lineage:
            metadata["success"] = True
            return None, metadata
        
        lineage_result, lineage_meta = _create_lineage_tracker(experiment, dataset_names)
        
        if not lineage_meta["success"]:
            logger.warning(f"Failed to create lineage tracker: {lineage_meta.get('error', 'Unknown error')}")
            metadata.update({k: v for k, v in lineage_meta.items() if k != "timestamp"})
            return None, metadata
        
        metadata["success"] = True
        return lineage_result, metadata
    except Exception as e:
        logger.error(f"Failed to set up lineage tracking: {str(e)}")
        return None, _create_error_metadata(e)


def _discover_and_validate_files(
    dataset_names: List[str],
    config: Dict[str, Any],
    lineage: Optional[LineageTracker]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Discover files matching the datasets and validate them.
    
    Args:
        dataset_names: List of dataset names to find files for
        config: The configuration dictionary
        lineage: Optional lineage tracker
        
    Returns:
        Tuple of (file_items, metadata)
    """
    try:
        metadata = _create_metadata()
        
        # Discover files matching the configuration
        discover_result, discover_meta = _discover_experiment_files(dataset_names, config, lineage)
        
        if not discover_meta["success"]:
            metadata.update({k: v for k, v in discover_meta.items() if k != "timestamp"})
            return [], metadata
        
        file_items = discover_result
        
        # Validate file items
        if not isinstance(file_items, list):
            metadata["error"] = f"File items must be a list, got {type(file_items).__name__}"
            return [], metadata
        
        if not file_items:
            logger.warning("No files found matching the configuration")
            metadata["success"] = True
            metadata["warning"] = "No matching files found"
            return [], metadata
        
        # Additional validation of file items structure
        for i, item in enumerate(file_items):
            if not isinstance(item, dict):
                metadata["error"] = f"File item at index {i} must be a dictionary, got {type(item).__name__}"
                return [], metadata
            
            # Validate required fields
            for field in ["path", "dataset"]:
                if field not in item:
                    metadata["error"] = f"File item at index {i} missing required field '{field}'"
                    return [], metadata
        
        metadata["success"] = True
        return file_items, metadata
    except Exception as e:
        logger.error(f"Failed to discover and validate files: {str(e)}")
        return [], _create_error_metadata(e)


def _load_and_process_files(
    file_items: List[Dict[str, Any]],
    schema: Optional[Dict[str, Any]],
    lineage: Optional[LineageTracker],
    track_lineage: bool
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load files into DataFrames and combine them.
    
    Args:
        file_items: List of file items to load
        schema: Optional schema to validate against
        lineage: Optional lineage tracker
        track_lineage: Whether to track lineage
        
    Returns:
        Tuple of (combined_df, metadata)
    """
    try:
        metadata = _create_metadata()
        
        # Load each file into a DataFrame
        load_result, load_meta = _load_all_files(file_items, schema, lineage, track_lineage)
        
        if not load_meta["success"]:
            metadata.update({k: v for k, v in load_meta.items() if k != "timestamp"})
            return pd.DataFrame(), metadata
        
        all_dataframes, processing_metadata = load_result
        
        # Validate loaded data
        if not isinstance(all_dataframes, list):
            metadata["error"] = f"Expected list of DataFrames, got {type(all_dataframes).__name__}"
            return pd.DataFrame(), metadata
        
        if not all_dataframes:
            logger.warning("No valid data loaded from any files")
            metadata["success"] = True
            metadata["warning"] = "No valid data loaded"
            return pd.DataFrame(), metadata
        
        # Combine all DataFrames
        combine_result, combine_meta = combine_dataframes(all_dataframes, lineage)
        
        if not combine_meta["success"]:
            metadata.update({k: v for k, v in combine_meta.items() if k != "timestamp"})
            return pd.DataFrame(), metadata
        
        result_df = combine_result
        
        # Validate the final DataFrame
        if not isinstance(result_df, pd.DataFrame):
            metadata["error"] = f"Expected DataFrame, got {type(result_df).__name__}"
            return pd.DataFrame(), metadata
        
        if len(result_df) == 0:
            logger.warning("Combined DataFrame is empty")
            metadata["success"] = True
            metadata["warning"] = "Empty result DataFrame"
            return result_df, metadata
        
        metadata["success"] = True
        metadata["rows_loaded"] = len(result_df)
        metadata["files_loaded"] = len(all_dataframes)
        logger.info(f"Successfully combined {len(all_dataframes)} DataFrames into one with {len(result_df)} rows")
        return result_df, metadata
    except Exception as e:
        logger.error(f"Failed to load and process files: {str(e)}")
        return pd.DataFrame(), _create_error_metadata(e)


def _export_lineage(
    lineage: LineageTracker,
    lineage_export_path: PathLike
) -> Dict[str, Any]:
    """
    Export lineage information to a file.
    
    Args:
        lineage: The lineage tracker
        lineage_export_path: Path to export to
        
    Returns:
        Metadata dictionary
    """
    try:
        metadata = _create_metadata()
        
        lineage_export_path = ensure_path(lineage_export_path)
        ensure_path_exists(lineage_export_path.parent)
        lineage.save_to_file(lineage_export_path)
        metadata["success"] = True
        logger.info(f"Exported lineage to {lineage_export_path}")
        return metadata
    except Exception as e:
        logger.error(f"Failed to export lineage: {str(e)}")
        return _create_error_metadata(e)


def load_from_file_list(
    file_paths: List[PathLike],
    schema: Optional[Dict[str, Any]] = None,
    track_lineage: bool = True,
    lineage_export_path: Optional[PathLike] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Load data directly from a list of file paths.
    
    This is a simplified entry point that bypasses configuration and discovery,
    allowing direct loading from file paths.
    
    Args:
        file_paths: List of paths to load
        schema: Optional schema for validation
        track_lineage: Whether to track lineage
        lineage_export_path: Optional path to export lineage
        metadata: Optional metadata to attach to all files
        
    Returns:
        Tuple containing (DataFrame, metadata_dict) with the loaded data and processing info.
        If processing fails, DataFrame will be None and metadata_dict will contain error details.
    """
    try:
        # Create lineage tracker if requested
        lineage = None
        if track_lineage:
            lineage_result, lineage_meta = create_lineage_tracker(
                name="FileListLoader",
                config={}
            )
            
            if not lineage_meta["success"]:
                logger.warning(f"Failed to create lineage tracker: {lineage_meta.get('error', 'Unknown error')}")
                # Continue without lineage
            else:
                lineage = lineage_result
                lineage.add_step("init", "Loading from explicit file list", {
                    "file_count": len(file_paths)
                })
        
        # Convert paths to item dictionaries
        file_items = []
        for path in file_paths:
            path_obj = ensure_path(path)
            item = {
                'path': path_obj,
                'file_name': path_obj.name
            }
            # Add additional metadata if provided
            if metadata:
                item.update(metadata)
            file_items.append(item)
        
        # Load each file
        load_result, load_meta = _load_all_files(file_items, schema, lineage, track_lineage)
        
        if not load_meta["success"]:
            return _empty_result(lineage, track_lineage, lineage_export_path), load_meta
        
        all_dataframes, processing_metadata = load_result
        
        if not all_dataframes:
            logger.warning("No valid data loaded from any files in the list")
            empty_result = _empty_result(lineage, track_lineage, lineage_export_path)
            return empty_result, {"timestamp": datetime.now().isoformat(), "success": True, "warning": "No valid data loaded"}
        
        # Combine all DataFrames
        combine_result, combine_meta = combine_dataframes(all_dataframes, lineage)
        
        if not combine_meta["success"]:
            return _empty_result(lineage, track_lineage, lineage_export_path), combine_meta
        
        result_df = combine_result
        
        # Export lineage if requested
        if lineage and lineage_export_path:
            try:
                lineage_export_path = ensure_path(lineage_export_path)
                ensure_path_exists(lineage_export_path.parent)
                lineage.save_to_file(lineage_export_path)
            except Exception as e:
                logger.warning(f"Failed to export lineage: {str(e)}")
                # Continue without failing the whole pipeline
        
        logger.info(f"Loaded {len(result_df)} rows of data from {len(all_dataframes)} files")
        return result_df, {"timestamp": datetime.now().isoformat(), "success": True, "rows_loaded": len(result_df), "files_loaded": len(all_dataframes)}
    except Exception as e:
        logger.error(f"Failed to load from file list: {str(e)}")
        return None, _create_error_metadata(e)


def _create_lineage_tracker(
    experiment: Dict[str, Any],
    dataset_names: List[str]
) -> LineageTracker:
    """Create a lineage tracker for experiment loading."""
    try:
        lineage = create_lineage_tracker(
            name=f"Experiment: {experiment.get('name', 'unnamed')}",
            config=experiment
        )
        lineage.add_step("init", "Initialized data loading process", {
            "experiment": experiment.get("name", "unnamed"),
            "datasets": dataset_names
        })
        return lineage
    except Exception as e:
        logger.error(f"Failed to create lineage tracker: {str(e)}")
        return None


def _discover_experiment_files(
    dataset_names: List[str],
    config: Dict[str, Any],
    lineage: Optional[LineageTracker]
) -> List[Dict[str, Any]]:
    """Discover all files matching the datasets in the configuration."""
    try:
        all_items = []
        
        for dataset_name in dataset_names:
            # Get dataset configuration
            dataset_config = config.get("datasets", {}).get(dataset_name)
            
            if dataset_config is None:
                logger.warning(f"Dataset '{dataset_name}' not found in configuration")
                continue
            
            # Discover files for this dataset
            discover_result, discover_meta = discover_dataset_files(dataset_name, dataset_config, config)
            
            # Skip this dataset if discovery failed
            if not discover_meta["success"]:
                logger.warning(f"Error discovering files for dataset '{dataset_name}': {discover_meta.get('error', 'Unknown error')}")
                continue
                
            items = discover_result
            
            # Record in lineage
            if lineage:
                lineage.add_step(
                    "discover_files",
                    f"Discovered files for dataset {dataset_name}",
                    {"dataset": dataset_name, "file_count": len(items)}
                )
            
            all_items.extend(items)
        
        return all_items
    except Exception as e:
        logger.error(f"Failed to discover experiment files: {str(e)}")
        return []


def _load_all_files(
    file_items: List[Dict[str, Any]],
    schema: Optional[Dict[str, Any]],
    lineage: Optional[LineageTracker],
    track_lineage: bool
) -> Tuple[List[pd.DataFrame], Dict[str, Any]]:
    """
    Load all files into DataFrames.
    
    This function processes each file item and loads it into a DataFrame,
    collecting processing metadata for each file.
    """
    try:
        metadata = _create_metadata()
        
        all_dataframes = []
        processing_metadata = {}
        
        for item in file_items:
            path_str = str(item['path'])
            logger.debug(f"Loading file: {path_str}")
            
            # Load the file
            df, file_metadata = load_file_into_dataframe(
                item, 
                schema=schema, 
                lineage=lineage, 
                track_lineage=track_lineage
            )
            
            # Process the result
            success = file_metadata.get('success', False)
            
            if success and df is not None and not df.empty:
                all_dataframes.append(df)
                
                # Collect metadata for this file
                processing_metadata[path_str] = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'success': True,
                    'timestamp': file_metadata.get('timestamp', '')
                }
            else:
                error_msg = file_metadata.get('error', 'Unknown error')
                logger.warning(f"Failed to load file: {path_str} - {error_msg}")
                processing_metadata[path_str] = {
                    'success': False,
                    'error': error_msg,
                    'error_type': file_metadata.get('error_type', 'Unknown')
                }
        
        metadata["success"] = True
        return all_dataframes, metadata
    except Exception as e:
        logger.error(f"Failed to load all files: {str(e)}")
        return [], _create_error_metadata(e)


def _empty_result(
    lineage: Optional[LineageTracker],
    track_lineage: bool,
    lineage_export_path: Optional[PathLike]
) -> pd.DataFrame:
    """Create an empty result DataFrame with proper lineage."""
    try:
        # Create empty DataFrame
        empty_df = pd.DataFrame()
        
        # Export lineage if requested
        if lineage and track_lineage:
            lineage.add_step("empty_result", "No data found matching criteria")
            
            if lineage_export_path:
                try:
                    lineage_export_path = ensure_path(lineage_export_path)
                    ensure_path_exists(lineage_export_path.parent)
                    lineage.save_to_file(lineage_export_path)
                except Exception as e:
                    logger.warning(f"Failed to export lineage: {str(e)}")
            
            # Attach lineage to the empty DataFrame
            try:
                empty_df = lineage.attach_to_dataframe(empty_df)
            except Exception as e:
                logger.warning(f"Failed to attach lineage to DataFrame: {str(e)}")
        
        return empty_df
    except Exception as e:
        logger.error(f"Failed to create empty result: {str(e)}")
        return pd.DataFrame()