"""
data_pipeline.py - Complete loading pipeline for fly rig data.

This module provides high-level functions that orchestrate the entire process
of loading fly rig data, from configuration loading to returning a merged DataFrame.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
from pathlib import Path
from loguru import logger

from ..config_utils.config_loader import load_merged_config
from ..config_utils.filter import filter_config_by_experiment
from ..discovery.files import discover_dataset_files
from ..schema.validator import get_schema_by_name
from ..lineage.tracker import LineageTracker
from .assembly import load_file_into_dataframe, combine_dataframes


def load_experiment_data(
    config: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
    *,
    track_lineage: bool = True,
    lineage_export_path: Optional[str] = None,
    experiment_name: Optional[str] = None,
    dataset_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load all data for an experiment based on configuration.
    
    This is the primary entry point for loading fly rig data. It orchestrates
    the entire process from configuration to returning a processed DataFrame.
    
    Args:
        config: Configuration dictionary (can be from load_merged_config)
        schema: Optional schema for validation (uses config schema if None)
        track_lineage: Whether to track data lineage
        lineage_export_path: Optional path to export lineage information
        experiment_name: Optional name to filter for a specific experiment 
        dataset_names: Optional list to filter for specific datasets
        
    Returns:
        DataFrame containing all experiment data with optional lineage tracking
    """
    # Filter for experiment if specified
    if experiment_name is not None:
        config = filter_config_by_experiment(config, experiment_name)
    
    # Extract experiment from config
    experiment = config.get("experiment", {})
    
    # Determine datasets to load
    if dataset_names is None:
        dataset_names = experiment.get("datasets", [])
        
    if not dataset_names:
        logger.warning("No datasets specified for loading")
        return pd.DataFrame()
    
    # Get schema if not provided
    if schema is None:
        schema_name = experiment.get("schema")
        if schema_name:
            schema = get_schema_by_name(schema_name, config=config)
    
    # Create lineage tracker if requested
    lineage = None
    if track_lineage:
        lineage = _create_lineage_tracker(experiment, dataset_names)
    
    # Discover files matching the configuration
    file_items = _discover_experiment_files(dataset_names, config, lineage)
    
    if not file_items:
        logger.warning("No files found matching the configuration")
        return _empty_result(lineage, track_lineage, lineage_export_path)
    
    # Load each file into a DataFrame
    all_dataframes, processing_metadata = _load_all_files(file_items, schema, lineage, track_lineage)
    
    if not all_dataframes:
        logger.warning("No valid data loaded from any files")
        return _empty_result(lineage, track_lineage, lineage_export_path)
    
    # Combine all DataFrames
    result_df = combine_dataframes(all_dataframes, lineage)
    
    # Export lineage if requested
    if lineage and lineage_export_path:
        lineage.save_to_file(lineage_export_path)
        logger.info(f"Exported lineage to {lineage_export_path}")
    
    logger.info(f"Loaded {len(result_df)} rows of data from {len(all_dataframes)} files")
    return result_df


def load_from_file_list(
    file_paths: List[Union[str, Path]],
    schema: Optional[Dict[str, Any]] = None,
    track_lineage: bool = True,
    lineage_export_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
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
        DataFrame containing all loaded data
    """
    # Create lineage tracker if requested
    lineage = None
    if track_lineage:
        lineage = LineageTracker(name="FileListLoader")
        lineage.add_step("init", "Loading from explicit file list", {
            "file_count": len(file_paths)
        })
    
    # Convert paths to item dictionaries
    file_items = []
    for path in file_paths:
        item = {
            'path': Path(path),
            'file_name': Path(path).name
        }
        # Add additional metadata if provided
        if metadata:
            item.update(metadata)
        file_items.append(item)
    
    # Load each file
    all_dataframes, processing_metadata = _load_all_files(file_items, schema, lineage, track_lineage)
    
    if not all_dataframes:
        logger.warning("No valid data loaded from any files in the list")
        return _empty_result(lineage, track_lineage, lineage_export_path)
    
    # Combine all DataFrames
    result_df = combine_dataframes(all_dataframes, lineage)
    
    # Export lineage if requested
    if lineage and lineage_export_path:
        lineage.save_to_file(lineage_export_path)
    
    logger.info(f"Loaded {len(result_df)} rows of data from {len(all_dataframes)} files")
    return result_df


def _create_lineage_tracker(
    experiment: Dict[str, Any],
    dataset_names: List[str]
) -> LineageTracker:
    """Create a lineage tracker for experiment loading."""
    lineage = LineageTracker(
        name=f"Experiment: {experiment.get('name', 'unnamed')}",
        config=experiment
    )
    lineage.add_step("init", "Initialized data loading process", {
        "experiment": experiment.get("name", "unnamed"),
        "datasets": dataset_names
    })
    return lineage


def _discover_experiment_files(
    dataset_names: List[str],
    config: Dict[str, Any],
    lineage: Optional[LineageTracker]
) -> List[Dict[str, Any]]:
    """Discover all files matching the datasets in the configuration."""
    all_items = []
    
    for dataset_name in dataset_names:
        # Get dataset configuration
        dataset_config = config.get("datasets", {}).get(dataset_name)
        
        if dataset_config is None:
            logger.warning(f"Dataset '{dataset_name}' not found in configuration")
            continue
        
        # Discover files for this dataset
        items = discover_dataset_files(dataset_name, dataset_config, config)
        
        # Record in lineage
        if lineage:
            lineage.add_step(
                "discover_files",
                f"Discovered files for dataset {dataset_name}",
                {"dataset": dataset_name, "file_count": len(items)}
            )
        
        all_items.extend(items)
    
    return all_items


def _load_all_files(
    file_items: List[Dict[str, Any]],
    schema: Optional[Dict[str, Any]],
    lineage: Optional[LineageTracker],
    track_lineage: bool
) -> Tuple[List[pd.DataFrame], List[Dict[str, Any]]]:
    """Load all files into DataFrames."""
    all_dataframes = []
    processing_metadata = []
    
    for item in file_items:
        # Load the file
        df, meta = load_file_into_dataframe(
            item,
            schema=schema,
            lineage=lineage,
            track_lineage=track_lineage
        )
        
        # Store metadata
        processing_metadata.append(meta)
        
        # Skip if loading failed
        if df is None or df.empty:
            logger.warning(f"No valid data loaded from {item['path']}")
            continue
        
        all_dataframes.append(df)
    
    return all_dataframes, processing_metadata


def _empty_result(
    lineage: Optional[LineageTracker],
    track_lineage: bool,
    lineage_export_path: Optional[str]
) -> pd.DataFrame:
    """Create an empty result DataFrame with proper lineage."""
    empty_df = pd.DataFrame()
    
    if lineage and track_lineage:
        lineage.add_step("warning", "No data loaded", {})
        
        # Attach lineage to DataFrame
        from ..lineage.tracker import attach_lineage_to_dataframe
        empty_df = attach_lineage_to_dataframe(empty_df, lineage)
        
        # Export lineage if requested
        if lineage_export_path:
            lineage.save_to_file(lineage_export_path)
    
    return empty_df