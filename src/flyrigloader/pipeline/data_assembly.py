"""
Data assembly for fly rig experiments.

This module provides low-level functions for transforming raw file data into structured
DataFrames with proper typing, metadata, and lineage tracking. It handles the actual
data loading and transformation logic, while higher-level pipeline orchestration is 
handled by the data_pipeline module.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime
from ..core.utils import ensure_path, PathLike, ensure_1d
from ..schema.validator import apply_schema
from ..readers.pickle import read_pickle_any_format
from ..lineage.tracker import LineageTracker
from ..core.factories import create_minimal_lineage_tracker, create_lineage_tracker


def load_file_into_dataframe(
    item: Union[PathLike, Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
    schema: Optional[Dict[str, Any]] = None,
    lineage: Optional[LineageTracker] = None,
    track_lineage: bool = True
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Read a file and transform it into a standardized DataFrame with metadata.
    
    This function supports multiple calling patterns:
    1. Dict item with metadata: {'path': Path, 'rig': 'rig1', ...}
    2. Direct file path: 'data/experiment/file.pkl'
    
    Args:
        item: File path or metadata dictionary with 'path' key
        config: Optional configuration dictionary
        schema: Optional schema for validation and transformation
        lineage: Optional LineageTracker for data provenance
        track_lineage: Whether to track data lineage
        
    Returns:
        Tuple of (DataFrame, metadata_dict)
        If processing fails, DataFrame will be None and metadata_dict will contain 
        structured error information including success status, error message, 
        error type, and traceback.
    """
    # Initialize processing metadata
    processing_meta = {
        "timestamp": datetime.now().isoformat(),
        "success": False
    }

    try:
        # Handle different input types
        if isinstance(item, (str, PathLike)):
            path = ensure_path(item)
            item_dict = {
                'path': path,
                'file_path': str(path),
                'file_name': path.name,
                'date': datetime.now().strftime("%Y-%m-%d")
            }
        else:
            item_dict = item
            # Check if 'path' key exists in the dictionary
            if 'path' not in item_dict:
                raise ValueError("'path' key missing in item dictionary")
                
            path = ensure_path(item_dict['path'])
        
        # Update processing metadata with file info
        processing_meta["file_processed"] = str(path)

        # Create or use lineage tracker
        if track_lineage:
            if lineage is None:
                tracker = create_minimal_lineage_tracker(
                    source=path, 
                    description=f"Loading file {path.name}"
                )
                lineage = tracker._tracker

            # Record source in lineage
            lineage.add_source(path, metadata=item_dict)
            lineage.add_step("load_file", f"Loading data from {path.name}", {
                "item_metadata": item_dict
            })

        # Load data from file 
        obj = read_pickle_any_format(path)
        
        # If loading failed, return error info
        if obj is None:
            processing_meta["error"] = "File read successfully but returned None"
            return None, processing_meta
        
        # Convert to DataFrame if it's a dictionary
        if isinstance(obj, dict):
            df_result = _dict_to_dataframe(obj, item_dict, lineage)
            df = df_result
        elif isinstance(obj, pd.DataFrame):
            df = obj
            if lineage:
                lineage.add_step(
                    "use_dataframe",
                    "Using existing DataFrame",
                    {"original_columns": list(df.columns)}
                )
        else:
            # Unsupported type
            processing_meta["error"] = f"Unsupported data type: {type(obj)}"
            return None, processing_meta
            
        # Attach metadata
        df_with_meta, overwritten_cols = _attach_metadata(df, item_dict)
        df, overwritten_cols = df_with_meta, overwritten_cols
        
        # Update processing metadata
        processing_meta["rows_processed"] = len(df)
        if overwritten_cols:
            processing_meta["overwritten_columns"] = overwritten_cols
            
        # Add metadata attachment to lineage
        if lineage:
            metadata_info = {"metadata_added": {k: v for k, v in item_dict.items() if k != 'path'}}
            if overwritten_cols:
                metadata_info["overwritten_columns"] = overwritten_cols
                
            lineage.add_step(
                "assign_metadata",
                "Adding metadata columns",
                metadata_info
            )
        
        # Apply schema if provided
        if schema:
            # Transform DataFrame according to schema 
            df = apply_schema(df, schema)
            
            if lineage:
                lineage.add_step(
                    "apply_schema",
                    "Applying schema transformations",
                    {
                        "schema_version": schema.get("schema_version", "unknown"),
                        "schema_name": schema.get("schema_name", "unnamed"),
                        "columns_after_schema": list(df.columns)
                    }
                )
        elif lineage:
            lineage.add_step(
                "skip_schema",
                "No schema provided, skipping schema application",
                {"columns": list(df.columns)}
            )
            
        # Mark processing as successful
        processing_meta["success"] = True
        return df, processing_meta
    
    except Exception as e:
        processing_meta = {
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": str(e.__traceback__)
        }
        return None, processing_meta


def _dict_to_dataframe(
    data_dict: Dict[str, Any],
    metadata: Dict[str, Any],
    lineage: Optional[LineageTracker] = None
) -> pd.DataFrame:
    """
    Convert a raw data dictionary into a DataFrame.
    
    Args:
        data_dict: Raw dictionary containing data arrays
        metadata: Metadata dictionary with 'path', 'rig', etc.
        lineage: Optional lineage tracker
        
    Returns:
        DataFrame with basic structure
    """
    # Handle time array, assuming 't' is the standard time key
    if 't' not in data_dict:
        raise ValueError("Data dictionary must contain a 't' array")

    t = ensure_1d(data_dict['t'], 't')

    # Start with basic metadata
    result_dict = {
        't': t,
        'rig': metadata.get('rig', 'unknown'),
        'date': metadata.get('date', 'unknown'),
        'file_path': str(metadata['path'])
    }

    # Add standard data columns
    standard_cols = ['frame', 'stimulus', 'response', 'trial']
    for col in standard_cols:
        if col in data_dict:
            value = data_dict[col]
            result_dict[col] = ensure_1d(value, col)
            
    # Also handle signal if present
    if 'signal' in data_dict:
        result_dict['signal'] = ensure_1d(data_dict['signal'], 'signal')
    
    # Add any other columns that can be converted to 1D arrays
    for key, value in data_dict.items():
        if key not in result_dict and key != 'signal_disp':
            try:
                result_dict[key] = ensure_1d(value, key)
            except (ValueError, TypeError) as e:
                if lineage:
                    lineage.add_step(
                        "skip_column",
                        f"Skipping column {key}",
                        {"reason": str(e)}
                    )
    
    df = pd.DataFrame(result_dict)
    
    # Handle special case for signal_disp (2D array of signal traces)
    if 'signal_disp' in data_dict and len(t) > 0:
        signal_disp = data_dict['signal_disp']
        if signal_disp.ndim == 2:
            # If signal_disp is 2D, make sure it matches timepoints
            if signal_disp.shape[0] == len(t):
                df['signal_traces'] = signal_disp.tolist()
            elif signal_disp.shape[1] == len(t):
                df['signal_traces'] = signal_disp.T.tolist()
            elif lineage:
                lineage.add_step(
                    "skip_signal_disp",
                    "Signal_disp dimensions don't match time array",
                    {
                        "signal_disp_shape": signal_disp.shape,
                        "time_length": len(t)
                    }
                )
        elif lineage:
            lineage.add_step(
                "skip_signal_disp",
                "Signal_disp is not 2D",
                {
                    "signal_disp_shape": signal_disp.shape,
                    "time_length": len(t)
                }
            )
                    
    if lineage:
        lineage.add_step(
            "create_dataframe",
            "Created DataFrame from raw data",
            {
                "columns": list(df.columns),
                "rows": len(df)
            }
        )
                    
    return df


def _attach_metadata(
    df: pd.DataFrame,
    metadata: Dict[str, Any]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Attach metadata to a DataFrame as columns.
    
    Args:
        df: DataFrame to attach metadata to
        metadata: Dictionary of metadata to attach
        
    Returns:
        Tuple of (modified_dataframe, list_of_overwritten_columns)
    """
    overwritten_cols = []
    
    # Attach metadata as columns
    for key, value in metadata.items():
        # Skip the path itself
        if key == 'path':
            continue
            
        # Check if column already exists
        if key in df.columns:
            overwritten_cols.append(key)
            
        # Set the column value
        df[key] = value
    
    return df, overwritten_cols


def combine_dataframes(
    dataframes: List[pd.DataFrame],
    lineage: Optional[LineageTracker] = None
) -> pd.DataFrame:
    """
    Combine multiple DataFrames into a single result.
    
    Args:
        dataframes: List of DataFrames to combine
        lineage: Optional lineage tracker
        
    Returns:
        Combined DataFrame with merged lineage
    """
    if not dataframes:
        empty_df = pd.DataFrame()
        if lineage:
            lineage.add_step(
                "combine_dataframes",
                "No DataFrames to combine, returning empty DataFrame"
            )
            # Attach lineage to empty DataFrame
            from ..lineage.unified import attach_lineage_to_dataframe
            empty_df = attach_lineage_to_dataframe(empty_df, lineage)
        return empty_df
    elif len(dataframes) == 1:
        df = dataframes[0]
        if lineage:
            lineage.add_step(
                "combine_dataframes",
                "Only one DataFrame to combine, returning it directly",
                {"rows": len(df), "columns": list(df.columns)}
            )
        return df
    
    # Combine multiple DataFrames
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    if lineage:
        lineage.add_step(
            "combine_dataframes",
            f"Combined {len(dataframes)} DataFrames",
            {
                "input_dataframes": len(dataframes),
                "total_rows": len(combined_df),
                "columns": list(combined_df.columns)
            }
        )
        
        # Attach consolidated lineage to the combined DataFrame
        from ..lineage.unified import attach_lineage_to_dataframe
        combined_df = attach_lineage_to_dataframe(combined_df, lineage)
    
    return combined_df
